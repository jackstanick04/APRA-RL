# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working boundaries

- This is a read-only diagnostic tool. Do NOT modify, edit, or write to `environment.py`, `distinguished_agent.py`, or `driver.py` without explicit approval each time.
- Use for: debugging, running existing experiments, reporting results, explaining existing code, and mechanical/cosmetic cleanup only when explicitly requested.
- Do NOT design new logic, reward functions, or experiment structures — that decision-making stays with the project owner.

## Project overview

APRA-RL trains a reinforcement learning agent to bid optimally in an Ascending Price Reimbursement Auction (APRA), a multi-round ascending auction where the auctioneer partially reimburses bidders for price increases they cause. The research goal is to find bidding/reimbursement policies that maximize auctioneer revenue while modeling realistic bidder behavior (including strategic abstention). This is a small, actively-evolving research codebase (three Python files), not a package — there is no build system, test suite, or dependency manifest yet.

## Running the code

There's no entry-point script beyond the driver itself:

```bash
python3 driver.py
```

This runs the full training loop (`NUM_EPISODES` defined at the top of `driver.py`, currently 30000 episodes) and writes per-round diagnostics to `training_log.txt` (overwritten each run) for episodes past the exploration warmup. Final win rate / reward / revenue / hype averages print to stdout.

Dependencies are `numpy`, `gymnasium`, and `torch` — no `requirements.txt` exists yet, so install them manually if setting up a fresh environment.

There are no linter or test configs in this repo currently.

## Architecture

Three files, each with a single responsibility (see `apra_rl_architecture.svg` for the original diagram — note it refers to `main.py`/`agent.py`, which are now `driver.py`/`distinguished_agent.py`):

- **`driver.py`** — training loop orchestrator. Owns all hyperparameters (env config, agent config, training config) as module-level constants at the top of the file, instantiates `Apra_env` and `Distinguished_agent`, and runs the episode/round loop: get observation → agent chooses a discretized bid → env steps → agent stores transition and updates policy → periodic target-network sync and epsilon decay. Also owns diagnostic logging (win/reward/revenue/hype logs, `training_log.txt` writer).
- **`environment.py`** (`Apra_env`, `gymnasium.Env` subclass) — auction mechanics. `reset()` draws a shared "true value" per episode and perturbs it with signal noise to produce the agent's and each opponent's private signal. `step()` collects all bids for the round (agent bid from the action passed in, opponent bids from `opp_bids()`), determines the round's max bid and reimburses the price increase (with a random tie-breaker), computes `valuation()` (blends private signal with current max bid — round 0 is signal-only), and returns the agent's reward via `reward()`. Opponent bidding strategies are hardcoded per-opponent-index inside `opp_bids()`: opponents 1 and 2 use `round_one_overbid` (a fixed round-0-only sniping strategy keyed on a signal cutoff), and all others use `loss_weight_bidding` (bids valuation plus a boost that grows with consecutive-round loss streaks, abstains once behind and no longer worth chasing). A bid of `None` represents abstention; abstaining costs nothing but forgoes `bid_cost` and reimbursement, and only counts toward reward on the final round if it results in a win.
- **`distinguished_agent.py`** (`Distinguished_agent`, `QNetwork`) — DQN bidder. `QNetwork` is a hardcoded 2-hidden-layer (64-unit) MLP mapping the observation to per-discretized-bid Q-values. `Distinguished_agent` implements standard epsilon-greedy action selection, an experience replay buffer (`deque`), Bellman-target policy updates (`update_policy`), and a soft (Polyak, `tau=0.05`) target network update (`update_target`). All of these are no-ops until the replay buffer has at least `batch_pull_size` transitions.

### Observation / action shape

Observation is a 4-vector: `[agent_signal, max_bid, agent_is_max_bid_holder, standardized_round_num]`. The action space is discretized into `NUM_AVAILABLE_BIDS` (101) options in `driver.py`, converted from a Q-network output index to a `[0,1]` bid via `index / (NUM_AVAILABLE_BIDS - 1)`; a raw bid `< 0.01` is treated as an abstention (`None`) by the environment.

### Key invariants to preserve when editing

- The agent is always bidder/opponent index 0 in `all_bids` / `reimbursements`; opponents are indices 1..`num_opponents`.
- `environment.py` state split: auction-parameter attributes set in `__init__` are fixed for the object's lifetime; auction-state attributes (current round, max bid, loss streaks, etc.) are the only ones reset in `reset()`.
- Only the agent's reward is computed — opponents are hardcoded, non-learning strategies, not additional RL agents.
- `driver.py` line 61-62 hardcodes a debug override for `round_num == 1` (forces a fixed action rather than querying the agent) — this is temporary test scaffolding called out in-line, not standard behavior; be aware of it when reasoning about round-1 dynamics.

## Notes from in-code comments worth preserving

- `opp_bids()` in `environment.py` is flagged in comments as needing a bigger rework down the line (opponents should eventually "shade" bids rather than bid up to valuation).
- `REIMBURSEMENT_RATES` is called out in `driver.py` as "the most important variable in this research" — treat changes to it as scientifically significant, not just a tuning knob.
