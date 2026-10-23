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
