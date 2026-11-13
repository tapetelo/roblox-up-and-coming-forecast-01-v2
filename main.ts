// Roblox Up-and-Coming top-10 forecaster (Node 22, standard library only).
//
// Model: a diffused, drifted rank kernel, fitted and backtested on the practice
// history (see README.md). For each game on the chart at the last read:
//   mu = current rank + 2 (average downward drift over the window)
//        - 0.8 * min(ranks climbed over the last 12 hours, 12) (graded
//        momentum: a climber keeps climbing, smoothly — no cliff at 10 ranks;
//        a slider is pushed down the same way), clamped to [1, 30]
//   and its chance of sitting at place k is a two-scale normal kernel around mu:
//     0.5 * N(k; mu, 2.5 + 0.15k) + 0.5 * N(k; mu, 10 + 0.3k)
//   normalised per place to sum 0.80, leaving 0.20 unnamed for games not named
//   (new entrants, deep-chart climbers).
// The wide component is what the log score needs: the chart moves fast (median 6
// top-10 places change between consecutive reads), so a sharp forecast that puts
// 0.5 on one game per place scores the floor (-9.21) whenever it is wrong, while
// the spread forecast keeps a live probability on every plausible game.
// Age-out: a game only stays on the chart for 28 days from its creation. When
// the history carries details (the house recorder's reads), a game's weights are
// scaled by the fraction of the scored window it survives; games that age out
// before the window starts are dropped. With no details (archive reads) this
// step is skipped. The window is read from task.json, never hardcoded.
// Deterministic: no randomness, no network, no clock; the same input always
// writes the same bytes.
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// --- tuned constants (backtest mean -34.65 vs baseline -46.96 over 4 splits;
// control v1 was -35.49) ---
const DRIFT = 2.0; // ranks the average game slides down over the window
const CLIMB_WINDOW_MS = 12 * 3600 * 1000; // look back this far for a climb
const CLIMB_GAIN = 0.8; // mu shift per rank climbed over the last 12h (graded)
const CLIMB_CAP = 12; // max ranks of climb that count toward the shift
const MU_MAX = 30; // projected ranks above this are clamped (as backtested)
const SD1_BASE = 2.5; // narrow kernel scale at place 1
const SD1_PER_PLACE = 0.15;
const SD2_BASE = 10.0; // wide kernel scale at place 1
const SD2_PER_PLACE = 0.3;
const MIX = 0.5; // weight of the narrow kernel
const TARGET_SUM = 0.8; // per-place named probability; 0.2 stays unnamed
const MIN_P = 0.0005; // rows below this are dropped
const AGE_LIMIT_MS = 28 * 24 * 3600 * 1000; // chart only lists games < 28 days old

interface ChartRow {
  rank: number;
  universe_id: number;
}
interface Read {
  at: string;
  up_and_coming: ChartRow[];
  details?: Record<string, { created?: string } | null> | null;
}

function normPdf(x: number, mu: number, sd: number): number {
  const z = (x - mu) / sd;
  return Math.exp(-0.5 * z * z) / (sd * Math.sqrt(2 * Math.PI));
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  window: { from: string; until: string };
};
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '');
const reads = lines.map((l) => JSON.parse(l) as Read);
const places = task.places;
const last = reads[reads.length - 1]!;
const tEnd = Date.parse(last.at);
const windowFrom = Date.parse(task.window.from);
const windowUntil = Date.parse(task.window.until);
const windowMs = Math.max(windowUntil - windowFrom, 1);

// Rank history per game: (time, rank), oldest first.
const hist = new Map<number, Array<{ t: number; rank: number }>>();
for (const r of reads) {
  const t = Date.parse(r.at);
  for (const row of r.up_and_coming) {
    const list = hist.get(row.universe_id) ?? [];
    list.push({ t, rank: row.rank });
    hist.set(row.universe_id, list);
  }
}
// Latest details per game (recorder reads only; archive reads carry none).
const created = new Map<number, number>();
for (const r of reads) {
  if (!r.details) continue;
  for (const [key, d] of Object.entries(r.details)) {
    if (d && d.created) created.set(Number(key), Date.parse(d.created));
  }
}

const chartLen = last.up_and_coming.length;
// Projected rank (mu) and survival weight per game on the last chart.
interface Game {
  uid: number;
  mu: number;
  rank: number;
  survive: number;
}
const games: Game[] = [];
for (const row of [...last.up_and_coming].sort((a, b) => a.rank - b.rank)) {
  let mu = row.rank + DRIFT;
  const pts = hist.get(row.universe_id) ?? [];
  // Rank at least CLIMB_WINDOW_MS ago: the most recent point that old, else the earliest.
  let oldRank: number | null = null;
  for (let i = pts.length - 1; i >= 0; i--) {
    if (tEnd - pts[i]!.t >= CLIMB_WINDOW_MS) {
      oldRank = pts[i]!.rank;
      break;
    }
  }
  if (oldRank === null && pts.length > 0) oldRank = pts[0]!.rank;
  // Graded momentum (IDEA 1): shift mu smoothly by how far the game climbed
  // over the last 12h, capped at 12 ranks gained; a game that slid down is
  // pushed the other way by the same rule. Mirrors the backtested
  // implementation exactly (hermes_test.py: mu = rank + drift - 0.8*min(cl, 12),
  // clamp mu to [1, 30]).
  const climbed = oldRank !== null ? oldRank - row.rank : 0;
  mu = row.rank + DRIFT - CLIMB_GAIN * Math.min(climbed, CLIMB_CAP);
  mu = Math.max(1, Math.min(MU_MAX, mu));
  // Age-out: fraction of the window the game survives (1 when no creation date known).
  let survive = 1;
  const c = created.get(row.universe_id);
  if (c !== undefined) {
    const outAt = c + AGE_LIMIT_MS;
    survive = Math.max(0, Math.min(1, (outAt - windowFrom) / windowMs));
  }
  if (survive > 0) games.push({ uid: row.universe_id, mu, rank: row.rank, survive });
}
// Defensive fallback: if every game aged out of the window (near-impossible
// with a real history, but the forecast must still be a valid shape), use the
// last chart as-is instead of emitting an empty forecast.
if (games.length === 0) {
  for (const row of [...last.up_and_coming].sort((a, b) => a.rank - b.rank)) {
    games.push({ uid: row.universe_id, mu: Math.max(1, Math.min(chartLen, row.rank + DRIFT)), rank: row.rank, survive: 1 });
  }
}

const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
for (let k = 1; k <= places; k++) {
  const sd1 = SD1_BASE + SD1_PER_PLACE * k;
  const sd2 = SD2_BASE + SD2_PER_PLACE * k;
  let total = 0;
  const weights: Array<[number, number]> = [];
  for (const g of games) {
    const w = g.survive * (MIX * normPdf(k, g.mu, sd1) + (1 - MIX) * normPdf(k, g.mu, sd2));
    weights.push([g.uid, w]);
    total += w;
  }
  if (total <= 0) continue;
  for (const [uid, w] of weights) {
    const p = TARGET_SUM * (w / total);
    if (p >= MIN_P) probabilities.push({ universe_id: uid, place: k, p: Math.round(p * 1e6) / 1e6 });
  }
}

// top10: the ten smallest projected ranks, ties broken by current rank.
const top10 = [...games]
  .sort((a, b) => (a.mu !== b.mu ? a.mu - b.mu : a.rank - b.rank))
  .slice(0, places)
  .map((g) => g.uid);

writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top10 })}\n`);
