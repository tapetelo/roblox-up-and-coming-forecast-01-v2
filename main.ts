// Roblox Up-and-Coming top-10 forecaster.
//
// It reads the chart history the harness provides (task.json plus
// history.ndjson, under MAKEFUN_INPUT_DIR) and writes one forecast.json into
// MAKEFUN_OUTPUT_DIR. Node 22, standard library only. No network, no
// randomness, and no reading of the clock: identical input files always give
// identical output bytes.
//
// The forecast, in plain terms. Take the last chart read and project every
// game on it forward into the scored window:
//   * a typical game drifts down about 2 ranks;
//   * a game that has been climbing carries some of that momentum, graded
//     smoothly: every rank gained over the last 12 hours lifts the projection
//     by 0.8 ranks, counting at most 12 ranks of gain, and a game that slid
//     down is pushed the other way by the same rule. There is no cliff in
//     it: a 9-rank climb earns nearly as much as a 10-rank one. The
//     projection is clamped to ranks 1..30.
//   * at each place k, a game's chance is a fifty-fifty blend of two normal
//     curves centred on its projection: a narrow one (spread 2.5 + 0.15k) and
//     a wide one (spread 10 + 0.3k). The wide half is the whole point: the
//     chart churns fast, so a forecast that piles everything onto one game
//     per place eats the log-score floor (-9.21) whenever it guesses wrong,
//     while a spread forecast keeps a live probability on every plausible
//     game.
//   * each place's chances are scaled to add up to 0.80; the other 0.20 is
//     left unnamed for games this forecast does not name (new entrants,
//     climbers from deep in the chart). Rows below 0.0005 are dropped.
//   * a game only stays listed for 28 days after it was created. When the
//     history carries creation dates (the house recorder's reads do), each
//     game's weight is scaled by the share of the scored window it is still
//     alive for, and games already gone before the window starts are left
//     out. Archive reads carry no creation dates, so that step simply does
//     nothing there.
//   * the window itself comes from task.json, never from a literal in here.
// The top 10 is just the ten games with the smallest projected ranks.
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// --- figures the backtest settled on (mean -34.65 across four splits;
//     baseline -46.96; the v1 control was -35.49) ---
const AVG_SLIDE = 2.0; // ranks a typical game falls over the window
const LOOKBACK_MS = 12 * 3600 * 1000; // how far back a climb is measured
const MOMENTUM_PER_RANK = 0.8; // projected-rank gain per rank climbed
const MOMENTUM_LIMIT = 12; // ranks of climb that can count, at most
const MU_CEILING = 30; // projections are clamped to [1, 30]
const NARROW_BASE = 2.5; // narrow curve's spread at place 1
const NARROW_GROWTH = 0.15; // ...growing this much per place
const WIDE_BASE = 10.0; // wide curve's spread at place 1
const WIDE_GROWTH = 0.3; // ...growing this much per place
const NARROW_SHARE = 0.5; // weight of the narrow curve in the blend
const NAMED_TOTAL = 0.8; // named probability per place; 0.2 stays unnamed
const FLOOR_P = 0.0005; // rows smaller than this are dropped
const MAX_AGE_MS = 28 * 24 * 3600 * 1000; // games age off the chart after this

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

interface Row {
  rank: number;
  universe_id: number;
}

interface Snapshot {
  at: string;
  up_and_coming: Row[];
  details?: Record<string, { created?: string } | null> | null;
}

interface Projected {
  uid: number;
  mu: number;
  rank: number;
  survive: number;
}

// Normal density at v, for centre c and spread s.
function gauss(v: number, c: number, s: number): number {
  const z = (v - c) / s;
  return Math.exp(-0.5 * z * z) / (s * Math.sqrt(2 * Math.PI));
}

function main(): void {
  const task = JSON.parse(readFileSync(join(IN_DIR, 'task.json'), 'utf8')) as {
    places: number;
    output: string;
    window: { from: string; until: string };
  };
  const rawText = readFileSync(join(IN_DIR, 'history.ndjson'), 'utf8');
  const snapshots: Snapshot[] = [];
  for (const line of rawText.split('\n')) {
    if (line !== '') snapshots.push(JSON.parse(line) as Snapshot);
  }

  const places = task.places;
  const latest = snapshots[snapshots.length - 1]!;
  const closeTime = Date.parse(latest.at);
  const winStart = Date.parse(task.window.from);
  const winEnd = Date.parse(task.window.until);
  const winSpan = Math.max(winEnd - winStart, 1);

  // Every game's (time, rank) trail, oldest first.
  const trails = new Map<number, Array<{ t: number; rank: number }>>();
  for (const snap of snapshots) {
    const t = Date.parse(snap.at);
    for (const row of snap.up_and_coming) {
      const trail = trails.get(row.universe_id);
      if (trail === undefined) {
        trails.set(row.universe_id, [{ t, rank: row.rank }]);
      } else {
        trail.push({ t, rank: row.rank });
      }
    }
  }

  // Creation time per game, from the newest read that carries details.
  const born = new Map<number, number>();
  for (const snap of snapshots) {
    if (!snap.details) continue;
    for (const [key, info] of Object.entries(snap.details)) {
      if (info && info.created) born.set(Number(key), Date.parse(info.created));
    }
  }

  const chartSize = latest.up_and_coming.length;
  const ordered = [...latest.up_and_coming].sort((a, b) => a.rank - b.rank);

  const projected: Projected[] = [];
  for (const row of ordered) {
    const trail = trails.get(row.universe_id) ?? [];
    // The rank the game held at least 12h ago: the newest trail point that
    // old, or the very first point when the game has no point that old.
    let anchor = -1;
    for (let i = 0; i < trail.length; i++) {
      if (closeTime - trail[i]!.t >= LOOKBACK_MS) anchor = i;
    }
    const rankThen =
      anchor >= 0 ? trail[anchor]!.rank : trail.length > 0 ? trail[0]!.rank : null;
    // Graded momentum: shift the projection smoothly by how far the game
    // climbed over the last 12h, capped at 12 ranks of gain; a game that
    // slid down is pushed the other way by the same rule.
    const climbed = rankThen !== null ? rankThen - row.rank : 0;
    let mu = row.rank + AVG_SLIDE - MOMENTUM_PER_RANK * Math.min(climbed, MOMENTUM_LIMIT);
    mu = Math.max(1, Math.min(MU_CEILING, mu));
    // Survival: what share of the scored window the game is still alive for
    // (1 when no creation date is known).
    let survive = 1;
    const bornAt = born.get(row.universe_id);
    if (bornAt !== undefined) {
      const leavesAt = bornAt + MAX_AGE_MS;
      survive = Math.max(0, Math.min(1, (leavesAt - winStart) / winSpan));
    }
    if (survive > 0) {
      projected.push({ uid: row.universe_id, mu, rank: row.rank, survive });
    }
  }

  // If every game aged out before the window (essentially impossible with a
  // real history, but the output still has to be well formed), fall back to
  // the raw last chart instead of writing an empty forecast.
  if (projected.length === 0) {
    for (const row of ordered) {
      projected.push({
        uid: row.universe_id,
        mu: Math.max(1, Math.min(chartSize, row.rank + AVG_SLIDE)),
        rank: row.rank,
        survive: 1,
      });
    }
  }

  const rows: Array<{ universe_id: number; place: number; p: number }> = [];
  for (let k = 1; k <= places; k++) {
    const narrow = NARROW_BASE + NARROW_GROWTH * k;
    const wide = WIDE_BASE + WIDE_GROWTH * k;
    const weights: Array<[number, number]> = [];
    let total = 0;
    for (const g of projected) {
      const w =
        g.survive *
        (NARROW_SHARE * gauss(k, g.mu, narrow) +
          (1 - NARROW_SHARE) * gauss(k, g.mu, wide));
      weights.push([g.uid, w]);
      total += w;
    }
    if (total <= 0) continue;
    for (const [uid, w] of weights) {
      const p = NAMED_TOTAL * (w / total);
      if (p >= FLOOR_P) {
        rows.push({ universe_id: uid, place: k, p: Math.round(p * 1e6) / 1e6 });
      }
    }
  }

  // Top 10: the ten smallest projected ranks, ties broken by current rank
// (a game that is already higher on the chart wins the tie).
  const top10 = [...projected]
    .sort((a, b) => (a.mu !== b.mu ? a.mu - b.mu : a.rank - b.rank))
    .slice(0, places)
    .map((g) => g.uid);

  writeFileSync(join(OUT_DIR, task.output), `${JSON.stringify({ probabilities: rows, top10 })}\n`);
}

main();
