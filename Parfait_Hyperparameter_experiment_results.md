# Hyperparameter Experiments: Parfait Uhiriwe

**Environment:** `ALE/Boxing-v5`
**Focus:** learning rate and batch size, and how the two interact.

All 10 experiments use **seed 1** and **500,000 training steps**. Everything else is fixed at the group baseline: γ = 0.99, ε 1.0 → 0.01 over 100,000 steps, replay buffer 50,000, target network update every 1,000 steps, a training step every 4 environment steps, and 10,000 warm-up steps.

**Mean score** is the average reward over the last 20 training episodes. The score is (our punches − opponent's punches), from −100 to +100, and a random agent scores about 0. A full round lasts about 1,780 steps; shorter episodes mean our agent won by knock-out (100 punches).

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Mean score (last 20) | Observed behaviour |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 100,000 | **+42.1** | **Best run.** Flat around 0 during exploration, then a steady climb from about episode 100, passing +40 by episode 245 and holding there. First knock-outs around episode 135, and many later ones (episodes as short as about 1,000 steps). Same settings as Teddy's best run, but a different seed gave +42.1 instead of +60.3. |
| 2 | 5e-5 | 0.99 | 64 | 1.0 | 0.01 | 100,000 | +14.45 | Too slow. Dropped to about −11 around episode 70, then crawled upward and only reached +14 by the end, still rising. Very few knock-outs. The updates are too small to learn much in 500,000 steps. |
| 3 | 2e-4 | 0.99 | 64 | 1.0 | 0.01 | 100,000 | +28.65 | Started improving earlier than exp 1 (around episode 100) and reached +20 by episode 175, but then levelled off and only crept up to about +29. Faster start, lower final level. |
| 4 | 5e-4 | 0.99 | 64 | 1.0 | 0.01 | 100,000 | **−31.05** | **Worst run.** Collapsed from 0 to about −38 within 70 episodes and never recovered, staying around −30 for the rest of training. Worse than random play, so the agent learned a harmful habit. It lost every round on points (always about 1,780 steps). Best episode in the whole run was only +9. The learning rate is too large for stable learning. |
| 5 | 1e-4 | 0.99 | 128 | 1.0 | 0.01 | 100,000 | +25.6 | Learned earlier than exp 1 (rising from about episode 80) and reached +25 by episode 150, then plateaued between +25 and +30 for the last 150 episodes. About 20% slower to train (29.2 vs 24.6 minutes). |
| 6 | 1e-4 | 0.99 | 96 | 1.0 | 0.01 | 100,000 | +28.25 | Steady, gradual improvement with a small dip around episode 160, ending around +28 and still rising slightly. Between batch 64 and batch 128 in both speed and final score. |
| 7 | 2e-4 | 0.99 | 128 | 1.0 | 0.01 | 100,000 | +19.25 | Learned earliest of all runs (about +10 by episode 75, while still exploring), but then stayed noisy and stalled around +20. Peaked at about +28 near episode 275, then dropped back to +19. The bigger batch did not help the higher learning rate. |
| 8 | 5e-5 | 0.99 | 128 | 1.0 | 0.01 | 100,000 | +30.15 | Flat for the first 100 episodes, then rose sharply between episodes 120 and 170 to about +28 and held there. More than double exp 2's score with the same learning rate, so the bigger batch made up for the small learning rate. |
| 9 | 2e-4 | 0.99 | 32 | 1.0 | 0.01 | 100,000 | −3.75 | **Second worst.** Never learned to win. Fell to about −14 by episode 110, then swung between −14 and −3 for the rest of training without a lasting trend. No knock-outs (every round about 1,780 steps), and best episode only +13. A higher learning rate with a small, noisy batch made updates too unstable. (Trained on a laptop CPU, so it took 227 minutes instead of about 25.) |
| 10 | 1.5e-4 | 0.99 | 64 | 1.0 | 0.01 | 100,000 | +29.05 | Refinement test between exps 1 and 3. Steady climb from about episode 80, reaching +20 by episode 175 and peaking at about +36 around episode 265, then settling near +29. Knock-outs from about episode 120. Better than 2e-4 at the end of training, but clearly below 1e-4, so the best learning rate is not above 1e-4. |

## What I found

**Learning rate has a clear sweet spot at 1e-4.** With batch size 64, the scores rise and then fall: +14.45 at 5e-5, **+42.1 at 1e-4**, +28.65 at 2e-4, and −31.05 at 5e-4. Too small and the agent learns too slowly to get far in 500,000 steps; too large and each update overshoots, which in exp 4 drove the agent into a bad policy it never escaped.

**A bigger batch was not better at the best learning rate.** At 1e-4, batch 64 scored +42.1, batch 96 +28.25 and batch 128 +25.6. Bigger batches also made training slower without paying off.

**Batch size and learning rate interact.** Doubling the batch from 64 to 128 helped the small learning rate a lot (5e-5: +14.45 → +30.15), because each update averages over more experiences and so does more useful work. But it did not help the larger learning rate (2e-4: +28.65 → +19.25). I expected larger batches to tolerate larger learning rates, since their gradients are less noisy, but my results did not confirm that.

**A small batch makes a high learning rate worse.** At lr 2e-4, batch 32 scored −3.75, batch 64 +28.65 and batch 128 +19.25. A batch of 32 gives noisy gradient estimates, and large steps in noisy directions kept the agent from settling on a good policy. So small batches clearly hurt, but going above 64 did not help either.

**Learning rate also changes when learning starts.** Higher learning rates (exps 3 and 7) started improving earliest, but then levelled off at lower scores than exp 1.

## Best and worst configurations

- **Best:** exp 1 (lr 1e-4, batch 64), at +42.1, with the most knock-out wins.
- **Worst:** exp 4 (lr 5e-4, batch 64), at −31.05, worse than random play. Exp 9 (lr 2e-4, batch 32) was second worst at −3.75.

## Final configuration

**Learning rate 1e-4, batch size 64** (exp 1), with the rest at the group baseline.

1. **It scored highest** (+42.1) and won the most rounds by knock-out.
2. **Every nearby value was worse.** Lower learning rates learned too slowly (5e-5: +14.45), and higher ones levelled off lower or collapsed (1.5e-4: +29.05, 2e-4: +28.65, 5e-4: −31.05). Exp 10 tested whether the best value sat slightly above 1e-4, and it did not. Batch sizes above and below 64 were also worse.
3. **It was the fastest of the good runs** (24.6 minutes, against 26–29 minutes for batches 96 and 128).
4. **It is the same configuration that gave the group's best result** (Teddy's exp 6, +60.3, with seed 42). It came out on top with two different seeds, which makes it more trustworthy than a single lucky run.

## Limitation

Each configuration was trained once. My exp 1 used exactly the same settings as Teddy's best run but a different seed and scored +42.1 instead of +60.3, so differences of about 10 points between runs may be chance. The large effects (exp 4's collapse, the learning-rate sweet spot, the batch-128 boost at 5e-5) are clear enough to trust.
