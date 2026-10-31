# Learning log

Notes written while building the project, in the order things were done. Each step has
**what**, **why**, and **what to say in an interview**. Training steps come after you
choose the hyperparameters.

---

## Step 0. The big picture

**The order book.** An exchange keeps, for every instrument, a list of resting limit
orders: bids (people willing to buy) sorted from the highest price down, and asks (people
willing to sell) sorted from the lowest price up. "Level 1" is the best bid and best ask,
"level 2" the next price on each side, and so on. A snapshot of the top 10 levels is 40
numbers: for each level, ask price, ask volume, bid price, bid volume.

```
          price     volume
ask L3    100.03      4.0
ask L2    100.02      1.5
ask L1    100.01      0.7     <- best ask
          ------ spread ------
bid L1    100.00      2.1     <- best bid
bid L2     99.99      3.3
bid L3     99.98      0.4
mid-price = (100.01 + 100.00) / 2 = 100.005
```

**The question.** Given the last 100 snapshots, will the mid-price over the next k events
be higher, lower, or about the same? That is a 3-class classification problem. The
intuition for why the book should help: if far more volume is waiting on the bid than on
the ask (order-book *imbalance*), sellers will exhaust the ask first and the price tends to
tick up. Market makers and HFT firms use signals like this on horizons of milliseconds to
seconds.

**DeepLOB** (Zhang, Zohren & Roberts, Oxford-Man Institute, 2019) learns those signals
directly from the raw 100 x 40 "image" of the book with a CNN, then an Inception module,
then an LSTM. The paper reports macro F1 of about 83% on the FI-2010 benchmark at k = 10.

**The pipeline.**

```
FI-2010 (public benchmark)          Binance BTCUSDT / ETHUSDT (recorded ourselves)
            \                                     /
     windows of 100 snapshots x 40 numbers, labels down / flat / up at horizon k
                              |
     baselines: logistic regression, LightGBM   vs   DeepLOB (CNN + Inception + LSTM)
                              |
     classification metrics (accuracy, macro F1)  ->  trading backtest with fees
                              |
     export to ONNX, measure inference latency in microseconds
```

**What makes the project credible.** Not the F1 score. Two things: (1) checking what the
benchmark labels actually measure (step 2 found a big surprise), and (2) the gap between
"predicts the direction 60% of the time" and "makes money after paying the exchange".

---

## Step 1. The FI-2010 dataset (`src/deeplob/fi2010.py`)

**What it is.** Ntakaris et al. (2018): five Finnish stocks on NASDAQ Nordic, 10 trading
days in June 2010, about 395,000 order book events. One row per event (an order arriving,
being cancelled, or trading). We use the copy shipped with the DeepLOB authors' code:
auction periods removed, "decimal precision" normalisation, setup 2 (train on days 1-7,
test on days 8, 9 and 10, one file each).

**The file layout is unusual**, so the loader documents it:

- Stored transposed: 149 rows x N events.
- Rows 0-39 are the book, rows 40-143 hand-crafted features (DeepLOB doesn't use them),
  rows 144-148 labels for k = 10, 20, 30, 50, 100 events, coded 1 up, 2 flat, 3 down. We
  recode them as 0 down, 1 flat, 2 up.
- Each file is the five stocks *one after another* (stock 1 for all days, then stock 2,
  ...). The stocks trade at very different price levels, so the four boundaries are huge
  jumps in the mid-price, and `stock_segments` finds them. A window that straddles two
  stocks is nonsense (half of it is another company), so no window is allowed to cross a
  boundary.

**Windows.** The input for the prediction at event t is the 100 snapshots t-99 ... t. The
loader stores the book once and slices windows on the fly, because materialising all
254,750 training windows would take 254,750 x 100 x 40 x 4 bytes = 4 GB.

**Validation split.** The authors' notebook uses the last 20% of the training file as
validation. Because the file is ordered by stock, that is almost all of stock 5: it tests
"does the model work on a different stock", not "does it work on later data". We instead
take the last 20% of *each stock's* events, with a 100-event gap so that no training label
(which looks up to 100 events ahead) peeks into the validation period. Same idea as the
purged split in the Nifty project.

**Interview line.** "Before training I checked the file layout. The benchmark file is
sorted by stock, so the reference validation split was really a different stock, and
windows could cross from one stock into another. I fixed both."

---

## Step 2. Sanity checks on the labels (`scripts/01_check_fi2010.py`)

Results in `results/fi2010_checks.md`.

**Class balance.** At k = 10, 71% of test events are "flat". A model that always says flat
gets 71% accuracy, which is why the paper (and we) report **macro F1**: the F1 score of
each class averaged with equal weight, so ignoring the rare up/down classes is punished.

**Can the labels be recomputed?** The FI-2010 paper defines the label from the mean of the
next k mid-prices compared with the current one (DeepLOB's paper uses the next-k mean vs
the previous-k mean), with a threshold. Recomputing either from the mid-prices in the file
agrees with the stored label only 50-69% of the time, even with the threshold chosen to
match the class shares. So the labels were not produced by either textbook formula from
these rows, at least not in a way we can reproduce.

**The surprise: how much of the label is already known?** Correlating the stored label
with price changes at every lag from -150 to +150 events, the strongest relationship is
with the *past* change over the last 1-9 events (correlation 0.6-0.75 on test day 10), not the future.
So we fitted a logistic regression on nothing but 8 past mid-price returns, all inside the
input window the model sees anyway:

| k | macro F1, past returns only | DeepLOB paper |
|---|---|---|
| 10 | 0.799 | 0.834 |
| 20 | 0.689 | 0.728 |
| 50 | 0.789 | 0.804 |

Eight numbers and a linear model come within 1.5-4 F1 points of the paper's deep network.
Real future price moves are close to unpredictable, so this means the FI-2010 labels
largely describe movement that has *already happened* by the time of the prediction. (Part
of this is by design: a label that compares the next-k mean with the previous-k mean
contains the recent past. But the effect here is larger than that formula gives.)

**What this means for the project.**

1. FI-2010 numbers are only useful for checking our DeepLOB matches the published one. We
   will report them next to the "past returns only" baseline, so the reader sees how much
   is genuinely learned.
2. For Binance, we define labels on the *future only*: the mean of the next k mids versus
   the mid at the moment of the prediction. Nothing in the label is known at prediction
   time.
3. The real test is the trading backtest, which can't be fooled by label definitions.

**Interview line.** "A linear model on eight past returns gets macro F1 0.80 on FI-2010 at
k = 10, close to DeepLOB's 0.83, because the benchmark's labels mostly encode price moves
that already happened. So I treated FI-2010 as a reproduction check and built my own
labels on crypto data using only future prices."

---

## Step 3. Recording Binance order books (`src/deeplob/recorder.py`)

**Where the data comes from.** Binance publishes a partial-depth WebSocket stream
`btcusdt@depth20@100ms`: the top 20 levels each side, every 100 ms. We keep the top 10.

**Two practical problems we hit.**

- *Geo-blocking.* GitHub's runners are in the US, and `api.binance.com` refuses US IPs
  (HTTP 451). Binance runs a public market-data mirror, `data-stream.binance.vision`, which
  serves the same streams without the restriction, so the recorder uses that. (Bybit's REST API was
  blocked too; OKX, Coinbase and Kraken worked. The probe is in the git history.)
- *Size.* 10 snapshots per second x 40 numbers x 2 symbols is 70 million numbers a day.
  Stored naively as floats that's hundreds of MB a day, too much for a git repo.

**How it's stored.** Prices as integer ticks (BTCUSDT tick = 0.01) and quantities as
integer lot steps, so nothing is lost to float rounding. Then each column is
*delta-encoded* over time: between two snapshots 100 ms apart most levels don't change, so
the differences are mostly zeros, which compress extremely well. A 5-minute test run was
35 KB per symbol, so about 10 MB per symbol per day. `load_chunk` undoes the encoding, and
`tests/test_recorder.py` checks the round trip is exact.

**How it runs.** The `record-binance` workflow records for just under 6 hours (a job's
limit) and commits the hourly files to `data/binance/`. It's scheduled every 6 hours, so
runs chain into a continuous recording. Gaps happen (a runner starting late, a dropped
connection); the timestamps show them, and windows will never cross a gap.

**A first look at the data.** BTCUSDT trades around $81,800 with a tick of $0.01, and the
spread is one tick almost all the time: about 0.001 basis points. Binance's taker fee is
10 bps (7.5 with BNB). So a strategy that crosses the spread pays roughly 10,000 times the
spread in fees. Price moves over a few seconds are usually far smaller than that, which is
exactly why "accurate" is not the same as "profitable" here.

**Interview line.** "I recorded my own Level-2 data from Binance's WebSocket feed, stored
it exactly as integer ticks with delta encoding (about 10 MB a day per symbol), and ran
the recorder on a schedule in GitHub Actions."

---

## Step 4. The DeepLOB network (`src/deeplob/model.py`)

**Reading the 100 x 40 input as an image.** Columns are (ask price, ask volume, bid price,
bid volume) for levels 1-10, rows are time. The convolution kernels are shaped to respect
that structure:

| block | kernel | what it combines | width after |
|---|---|---|---|
| conv 1 | 1x2, stride 2 | each price with its volume | 20 |
| conv 2 | 1x2, stride 2 | the ask side with the bid side at each level | 10 |
| conv 3 | 1x10 | all 10 levels into one feature | 1 |

Each block also has two 4x1 convolutions along time (no padding, so 100 time steps shrink
to 82). Then an **Inception module** runs 3x1, 5x1 and max-pool branches in parallel (64
channels each), so the network sees several time scales at once. Finally an **LSTM** with
64 hidden units reads the 82 steps and its last state goes to a linear layer with 3
outputs. 143,907 parameters, which is small: the paper stresses it's fast enough for
real-time use.

**One fix to the reference code.** The authors' PyTorch notebook applies softmax at the
end of `forward` and then trains with `nn.CrossEntropyLoss`, which applies log-softmax
itself. Softmax twice squashes the outputs and weakens the gradients. Our model returns raw
logits.

**Tests.** Shapes, the parameter count, and that it can drive the loss to near zero on a
tiny batch (if a network can't overfit 16 examples, something is broken).

**Cost on this machine.** One pass over the 200,000 training windows takes about 18
minutes on 4 CPU cores, so a 100-epoch run like the paper's would take over a day. Choices
for that come with the hyperparameters.

---

## Step 5. Training on FI-2010 and choosing hyperparameters (`src/deeplob/train.py`)

Full table: `results/fi2010_summary.md` (made by `scripts/04_summarize_fi2010.py`).

**How training works.** Adam, cross-entropy loss, mini-batches of windows. After every
epoch we score the validation set (the last 20% of each stock's training days) and keep the
weights with the best validation macro F1. Training stops after 10 epochs without
improvement. To fit GitHub's 6-hour job limit on CPUs we train on every 4th window; next-door
windows share 99 of their 100 snapshots, so this throws away much less information than it
sounds like.

**Choosing hyperparameters without cheating.** You asked to try 2-3 settings and keep the
best. "Best" must be measured on validation, never on test: if you pick the setting with
the best *test* score, the test score is no longer an honest estimate of performance on new
data (the same trap as naive CV in the Nifty project). So each setting's test score is
computed once, and only the validation-chosen one is reported.

| setting | what it is | why try it |
|---|---|---|
| A | lr 1e-4, batch 64 | the authors' notebook |
| B | lr 1e-3, batch 128 | 10x faster learning |
| C | lr 1e-2, Adam eps 1, batch 32 | the paper's stated settings |
| D | B, with each stock's inputs z-scored by its own training statistics | added after A-C, to test whether input scaling explained the gap below |

Nine runs (A-C x k = 10, 20, 50) ran in parallel on GitHub Actions, about 1-2 hours each.
B won on validation at every horizon and A was clearly worst: lr 1e-4 is simply too slow
for this budget.

**Result: our DeepLOB is well below the paper and below simple baselines.**

| model (test macro F1) | k = 10 | k = 20 | k = 50 |
|---|---|---|---|
| Logistic regression, 8 past returns | 0.800 | 0.690 | 0.786 |
| LightGBM, 53 book features | 0.806 | 0.716 | 0.832 |
| DeepLOB, ours (chosen setting) | 0.663 (D) | 0.609 (B) | 0.737 (D) |
| DeepLOB, paper | 0.834 | 0.728 | 0.803 |

**Why, as far as we can tell.**

- *Not input scaling.* Setting D (per-stock normalisation) changed F1 by -0.03 to +0.01.
- *Overfitting.* Training loss keeps falling while validation loss starts rising after 10-25
  epochs: the network memorises the 7 training days. The F1 for the up and down classes is
  about 0.55-0.67 versus 0.75 for LightGBM; the network mostly learns "flat".
- *The task rewards a shortcut the network has to rediscover.* Step 2 showed the labels are
  mostly the recent past move. LightGBM and logistic regression are handed past returns as
  features; DeepLOB has to work them out from raw prices, where the moves are 10,000 times
  smaller than the price level.
- *Less training.* The paper trained much longer on a GPU with every window.

So our reproduction does not reach the published number. We report that as it is rather
than tuning on the test set until it does: three settings plus one diagnostic were tried,
all chosen on validation.

**Interview line.** "My DeepLOB reached macro F1 0.66 to 0.74 on FI-2010 against 0.80 to
0.83 in the paper, and a LightGBM on 53 hand-made features beat both. I picked
hyperparameters on validation only, checked that input scaling wasn't the cause, and
found the network overfits a 7-day training set on a benchmark whose labels mostly encode
past returns. That is why my main evaluation is on crypto data with future-only labels and
a fee-aware backtest."

---

## Step 6. The Binance experiment: labels, splits and baselines (`src/deeplob/binance.py`)

*Data: one week of recording, Oct 23 09:17 to Oct 30 19:07 UTC (172.6 hours, about 6
million snapshots per symbol). A first run on the first 31 hours is compared at the end.*

**Choosing the horizon from the data.** BTCUSDT's mid-price changes only about once every
5 seconds: the spread is one tick ($0.01 on $81,000, about 0.001 bp) and there are often
several BTC waiting at the best price. Over 1 second the mean move is 0.06 bp, so almost
every 1-second label would be noise. We predict 10, 30 and 60 seconds ahead (100, 300, 600
snapshots), from the last 10 seconds of book (100 snapshots), one prediction per second.

**Labels use only the future.** r = mean of the next h mids / current mid - 1. Unlike
FI-2010 (step 2), nothing in the label is known at prediction time. The flat band alpha is
set on the training period so that the three classes are equally common: 0.22 bp (BTC) and
0.65 bp (ETH) at 30 s. For BTC at 10 s the band is zero: in more than a third of training
windows the mid doesn't move at all in the next 10 seconds, so "flat" there means exactly
no change.

**Chronological split with an embargo.** First 60% of the recording for training (about 4.3
days), next 20% for validation, last 20% for test (about 34 hours, Oct 29 morning to Oct 30
evening), and `T + h` snapshots dropped at each cut so no label peeks across. Shuffling
snapshots randomly into train and test would put almost identical neighbours on both sides,
the same leakage as naive K-fold in the Nifty project.

**Inputs.** Prices as log distance from the window's last mid, divided by a per-column scale
fitted on training windows; volumes as log(1 + quantity), z-scored on training. So the model
sees the shape of the book and how it moved within the window, not the BTC price level.
BTC and ETH are pooled into one model per horizon.

**Results (test macro F1, chance = 0.333).**

| model | 10 s | 30 s | 60 s |
|---|---|---|---|
| Logistic regression, past returns only | 0.259 | 0.337 | 0.315 |
| Logistic regression, book features | 0.485 | **0.426** | **0.409** |
| LightGBM, book features | 0.466 | 0.372 | 0.345 |
| DeepLOB (setting chosen on validation) | **0.499** | 0.421 | 0.395 |

Four things to notice. Past returns alone are at or below chance: with honest labels there is
no momentum to exploit at these horizons, the opposite of FI-2010. (At 10 s that model mostly
predicts "flat", having learned that quiet seconds are followed by quiet seconds; the test
days were busier, so it scores below chance.)

The order book does carry information (0.49-0.50 at 10 s), which fades with the horizon, as
you'd expect: the book tells you about the next few seconds, not the next minute.

LightGBM is worse than the linear model at every horizon: it fits patterns of the training
days that don't carry over to later days.

And **DeepLOB is the best model only at 10 s**, by 1.4 points over logistic regression; at 30
and 60 s the linear model is 0.5-1.4 points ahead. On one 34-hour test period, gaps of one
point are within noise, so the honest summary is: DeepLOB matches a well-built linear model
on hand-made features, it doesn't beat it. DeepLOB also saw fewer examples: one window every
6 seconds (122,000 windows over both symbols) to fit GitHub's 6-hour CPU job limit, against one per second for
the baselines.

**DeepLOB settings tried** (you asked for 2-3, best on validation): B (lr 1e-3, batch 128,
the FI-2010 winner), C (the paper's lr 1e-2 with Adam eps 1) and E (lr 3e-4 with weight
decay, aimed at the overfitting seen on FI-2010). Nine jobs ran in parallel on GitHub
Actions, 25-120 minutes each. Validation picked E at 10 s and B at 30 and 60 s; the three
settings were within 1.6 points of each other everywhere (table in
`results/binance_summary.md`). At 30 and 60 s, C happened to do slightly better on test
than the chosen B (0.427 vs 0.421, 0.401 vs 0.395), but we report the validation choice:
picking on test would be the leak we avoided everywhere else. Most runs peaked after 5-6
epochs (range 2-12) and then overfit, even with 5.6 times as many hours of data as the first
run.

**What changed from the first 31 hours.** The first run (data to Oct 24 21:13, a 6-hour test
period) had DeepLOB best at every horizon: 0.541 / 0.466 / 0.435 against 0.524 / 0.450 /
0.407 for logistic regression. With the full week every model scored lower and DeepLOB's lead
went away. The main reason is the test days: the mid moved more often, so only 14-17% of
test labels are "flat" (a third or more in training), and every model is worst on that
class (at 10 s, flat F1 is 0.25-0.34 against 0.54-0.60 for up and down). The lesson: a 6-hour
test period was too short to rank the models, and the earlier "DeepLOB wins" didn't survive
more data. That is why we reran.

## Step 7. Does it make money? (`src/deeplob/backtest.py`)

**The rule.** Every h seconds: go long one unit if "up" is the most likely class, short if
"down", flat otherwise; hold for h seconds. Buy at the ask, sell at the bid, and pay the
exchange fee on notional each time the position changes. No queue position, latency or
market impact, so reality is worse than this.

**Result (one-week run, 34-hour test period).** With zero fees, all 18 model / symbol /
horizon combinations make money, with hit rates of 51.5-57.4% and 0.3-0.7 bp per trade. The
*break-even fee* (the fee per side at which profit is zero) is at most 0.69 bp. Binance
charges 10 bp per side at the base tier (7.5 with the BNB discount). So the strategy that is
"right 55% of the time" loses 14-29 times what it makes once you pay the fees.

| fee per side | net P&L, logistic regression, 10 s, BTC |
|---|---|
| 0 bp | +5,387 bp |
| 1 bp | -4,163 bp |
| 10 bp | -90,113 bp |

DeepLOB doesn't make more money than the simpler models: at 10 s it nets +5,617 bp on BTC at
zero fees (logistic regression +5,387, LightGBM +5,743) and +4,887 on ETH (logistic regression
+4,969), and its break-even fees, 0.43-0.59 bp, are in the middle of the pack. The first
31-hour run showed the same thing with DeepLOB's better F1: **better classification did not
mean better trading.** Macro F1 counts every correct call equally, but P&L depends on getting
the *large* moves right, and a model can gain F1 on small moves that don't pay. Full table:
`results/binance_summary.md`; the web app reruns it at any fee.

**What this means.** The signal is real but tiny compared to costs. Firms that trade on it
are market makers who *earn* the spread with limit orders and pay near-zero or negative
fees, or they use it to time execution of trades they were going to make anyway. A retail
taker can't monetise it. That is the gap between "55% accuracy" and "profitable after
fees" the project set out to show.

**Interview line.** "On a week of Binance BTC and ETH order books, the book predicts the
10-second mid-price direction with macro F1 0.50 against 0.33 for chance, out of sample, and
the signal makes about 0.5 bp per trade before costs. The break-even fee is under 0.7 bp;
Binance's taker fee is 10. A CNN-LSTM on the raw book matched logistic regression on
hand-made features but didn't beat it."


## Step 8. Speed: ONNX export and latency (`scripts/06_latency.py`)

**Why it matters.** A signal that predicts the next 10 seconds is only useful if the
prediction arrives well within those seconds. HFT firms care about microseconds; here the
question is just whether DeepLOB is fast enough to run on every book update (every 100 ms).

**What we did.** Exported the trained 10-second model (setting E) to ONNX (an open format
that fast inference engines read) and timed one prediction at a time on one CPU core, 2,000
calls.

| engine | median | p99 |
|---|---|---|
| PyTorch eager | 2,930 us | 5,843 us |
| ONNX Runtime | 943 us | 1,439 us |

ONNX Runtime is 3.1x faster, and its outputs match PyTorch to 1.2e-7. The first measurement
(Oct 25, a different cloud machine) gave 807 us median and 930 us p99, so the tail depends on
the machine more than on the model; on a shared cloud CPU, other jobs cause the slow calls.
At under 1 ms per prediction, one core can keep up with 100 ms updates about 100 times over.
For real HFT, the next steps would be computing the convolutions incrementally (each new
snapshot only changes the last row of the window), quantising to int8, or rewriting the
network in C++.

**Interview line.** "Exported to ONNX: about 0.9 ms median per prediction on one CPU core,
3x faster than PyTorch eager, with identical outputs."
