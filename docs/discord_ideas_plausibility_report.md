# Plausibility Report: Data-Addressable Topics in KQ Discord

Analysis of 108 chunks (~5,318 messages, Aug 2022 - Feb 2026) from the Killer Queen gameplay-general Discord channel. Topics are clustered by theme and all are plausible to answer with HiveMind event data.

---

### 1. Victory Condition Distribution by Map / Full Military Strategy

The single most discussed strategic topic in the entire Discord. Players debate whether full military (all warriors, no objective runner) is viable, on which maps, and whether objective play (berries/snail) reliably beats pure military.

- **Chunks**: 0-2, 9-17, 27-35, 36-37, 63-71, 90
- **Conversation bursts**:
  - `8/19/22 10:49 AM -- 4:38 PM` (initial queen playstyle / victory type debate)
  - `8/24/22 -- 9/16/22` (lockout map strategy, snail vs berry dominance)
  - `10/6/22 -- 10/19/22` (full mil on Day, snail riding vs military)
  - `12/12/23 -- 3/17/24` (Dusk/Day openers, full mil effectiveness revisited)
  - `6/6/25 3:03 PM -- 5:27 PM` (objective programs vs military meta, queen-weak era)
- **Total messages**: ~250+
- **Distinct users**: ~55+ (woodie, Prashant, JoshB, Jess [SF], BrianM [SF], Hel [SF], Galgo, Alacrity, CalculusHw, Le Chiffre, Elyse [SLC], Kevin J [PDX], Mario [DFW], richard [nyc], TJ [CHA], Felix [SF], lizz, loproto [SEA], Jeff [SLC], Blue Chris [SF], Marshall, rrenaud, tyler [nyc], Line Rider 0 [sf], and many more)
- **Key quotes**:
  - "Full mil is a transition state you use to get 3 speed warrior + speed drone" (JoshB)
  - "full mil is basically used to transition to exodia, and works best on day since obj is slow out the gate" (Galgo)
  - "the problem with it is that there is no way to 'bank' your advantage, outside of queen kills" (BrianM [SF])
  - "Objective programs reliably beat this [over-prioritized speed warriors]." (woodie)
  - "im really skeptical of berries being good on this map. I have a feeling snail will be dominant again" (Prashant)
- **Data approach**: Filter games by map. Compare victory type distribution (military/economic/snail) per map. Detect "full mil" phases by identifying periods where all 4 non-queen players have wings (all spawned as warriors via useMaiden events), specifically in the early/mid game (not late-game famine prep when berries are scarce). Compare win rates for games with early full-mil phases vs mixed compositions, stratified by map. Track how the meta has shifted over time.

---

### 2. Gate Control Correlation with Wins

A massive, contentious debate: does controlling more gates actually cause you to win, or is it just correlated? Players argue about whether gate control matters during famine, whether it's "circular," and what warrior count means for gate value.

- **Chunks**: 13-17, 21-22, 37, 94-97, 102-104
- **Conversation bursts**:
  - `9/7/22 -- 9/16/22` (gate positioning on lockout, queen vulnerability near gates)
  - `9/18/22 -- 10/4/22` (gate control stats, lockout mechanics)
  - `10/19/22 1:41 PM -- 2:16 PM` (lockout advantage banking)
  - `7/26/25 -- 8/3/25` (gate control during famine — 110-msg mega-thread)
  - `1/21/26 9:07 AM -- 11:45 AM` (gate control % vs win rate — 150-msg debate)
- **Total messages**: ~350+
- **Distinct users**: ~40+ (Asher, Marshall, woodie, Sky [nyc], Elyse [SLC], JoshB, Jess [SF], Blue Chris [SF], Prashant, BrianM [SF], Don [PDX], Le Chiffre, Ram, Jeff [SLC], nebunez, tyler [nyc], coppola [CHI], richard [nyc], rrenaud, CalculusHw, and more)
- **Key quotes**:
  - "gate control to win % always felt a bit circular to me...I had like 90% gate control and just lost" (Don)
  - "Attaining lockout on day is a soft-banked advantage. At least as banked as a good chunk of snail progress" (Jess [SF])
  - "gates fundamentally don't do anything during famine so in that sense gate control doesn't matter for the entire duration of famine" (Asher)
  - "1-3 warrior count during a normal game state is not much implied progress at all if you have gate control... It's a lot of implied progress during famine" (woodie)
- **Data approach**: For each game, estimate gate control from warrior spawn patterns (which gates warriors form at). Correlate gate control % at various game-time marks with eventual win/loss. Segment by: famine vs non-famine, warrior parity vs advantage, map, and score closeness. Run regression on gate% vs win probability.

---

### 3. Map-Specific Opening Strategies (Dusk, Day, Twilight)

Players intensely debate optimal openers: high-mid vs low-mid on Dusk, wrap opens on Twilight, 2-2 berry vs 3-1 snail splits, and how openers interact with map geometry.

- **Chunks**: 43, 65-67, 101
- **Conversation bursts**:
  - `1/27/23 10:00 AM -- 11:17 AM` (two-berry opening strategies on Dusk)
  - `1/11/24 -- 3/7/24` (Dusk high-mid vs low-mid, full mil on Day, 2-2 berry viability)
  - `1/15/26 -- 1/21/26` (Twilight wrap open vs speed gate strategies)
- **Total messages**: ~185+
- **Distinct users**: ~35+ (Le Chiffre, Elyse [SLC], Jeff [SLC], Carlitos, Ram, Prashant, JoshB, woodie, Marshall, Sky [nyc], Alacrity, rrenaud, TJ [CHA], Abby [bmore], Blue Chris [SF], nebunez, dee [nyc], CalculusHw, Mario [DFW], Showtime, alicemyslime [NYC], richard [nyc], and more)
- **Key quotes**:
  - "High mid open you can reach far gate before first vanilla, low mid you can't" (woodie)
  - "I think mid is always better and every team I've played on since 2018 has opened high mid" (Prashant)
  - "if you run two top berries in dusk open, you cede snail till past nub. its very not cheap" (Prashant)
  - "wrap open was really strong against enemy queen opening near speed" (Blue Chris)
- **Data approach**: Classify opening strategy from first 30 seconds of event data: early berryDeposit clustering = berry open, early playerKill clustering = military open, gate tag order = wrap vs direct. Compare win rates by opener type per map. Analyze mirror matchups vs asymmetric openers.

---

### 4. Snail Strategy, Duration, and Guard Positioning

A perennial topic: how long to ride snail, when to dismount, how speed warriors should cycle through to clear snail, and how vanguard/rearguard positioning affects snail progression.

- **Chunks**: 22-26, 27-35, 54, 60, 66, 84-85, 97
- **Conversation bursts**:
  - `10/3/22 -- 10/19/22` (snail riding duration, speed warrior cycling, enemy denial)
  - `3/3/23 -- 3/23/23` (speed obj on Night, snail position strategy)
  - `6/13/23 -- 7/20/23` (warrior roles relative to snail, inside/outside rotation)
  - `3/4/24 -- 3/7/24` (snail vs berry tradeoffs on Night)
  - `4/15/25 -- 5/6/25` (famine strategy, snail pushing during famine)
  - `7/31/25 -- 8/3/25` (vanguard/rearguard positioning, Night stage 1→2 transition)
- **Total messages**: ~170+
- **Distinct users**: ~30+ (JoshB, Elyse [SLC], Blue Chris [SF], Prashant, Galgo, Line Rider 0 [sf], Jess [SF], CalculusHw, woodie, rrenaud, Jc [CIN], dee [nyc], Leo [NYC], Le Chiffre, coppola [CHI], and more)
- **Key quotes**:
  - "riding the snail is gold here. makes so much pressure. even those few extra inches" (Prashant)
  - "Speed warriors should constantly be cycling through the snail and clearing it out" (woodie)
  - "The #1 spot where this community is really bad about establishing a vanguard is when snail moves from stage 1 to stage 2 on Night" (woodie)
  - "when it comes to moving snail... everything to do with whoever currently has warrior advantage" (woodie)
- **Data approach**: Aggregate getOnSnail/getOffSnail events per game. Compute ride duration, dismount frequency, snail position progress rate. Key analysis: detect "speed snail to speed warrior" transitions where a player rides snail with speed buff, dismounts, and then gets a playerKill shortly after (indicating they converted from snail rider to combat warrior). Measure frequency and win rate of this pattern vs teams that keep a dedicated snail rider. Analyze by map.

---

### 5. Team Composition States and Full Flex Meta

The community is evolving from simple W-D (warrior-drone) notation to SW-VW-SD-VD (speed warrior, vanilla warrior, speed drone, vanilla drone). Debate rages about whether "full flex" teams (everyone can play every role) are the future or impractical.

- **Chunks**: 23-25, 27-28, 46-47, 54-62
- **Conversation bursts**:
  - `10/4/22 -- 10/18/22` (full flex effectiveness, dedicated obj vs flex)
  - `1/31/23 -- 2/21/23` (full flex as meta, Bob strats team example)
  - `3/3/23 -- 12/12/23` (SW-VW-SD-VD notation, speed buff importance, team comp states)
- **Total messages**: ~115+
- **Distinct users**: ~30+ (JoshB, Jess [SF], CalculusHw, Elyse [SLC], Blue Chris [SF], Prashant, Alacrity, Le Chiffre, BrianM [SF], Felix [SF], Aquafox, woodie, Monopoli, Mario [DFW], TheAwfulDin (Ryan), dee [nyc], loproto [SEA], Jc [CIN], lizz, Lex, Line Rider 0 [sf], and more)
- **Key quotes**:
  - "Increasingly it's becoming more important to consider how many speed buffs each team has" (JoshB)
  - "full flex is TAS only, the communication alone adds so much difficulty" (CalculusHw)
  - "the best implementation of a full flex team to date was the Bob strats team" (Alacrity)
- **Data approach**: From useMaiden events (maiden_speed vs maiden_warrior), classify each player's buff type per game. Build SW-VW-SD-VD state vectors at key game moments. Correlate composition states with victory type and win rate. Measure "flex" via entropy: compute how evenly snail movement, berry deposits, and speed warrior time are distributed across drones on a team. High entropy = full flex (everyone does everything), low entropy = dedicated roles. Correlate flex entropy with win rate to test whether full-flex teams actually outperform specialist teams.

---

### 6. Queen Playstyle, K/D, and the "Queen-Weak Era"

Players debate power queen vs facilitatory queen, whether we're in a "queen-weak era" where warriors kill queens too easily, and whether queen focus should be gate control or warrior elimination.

- **Chunks**: 0-2, 90-91, 99
- **Conversation bursts**:
  - `8/19/22 10:49 AM -- 4:38 PM` (power queen vs facilitatory queen, NYC playstyle)
  - `6/6/25 -- 7/16/25` (queen-weak era claim, warrior K/D dominance)
  - `8/7/25 -- 10/17/25` (gate control priority vs warrior elimination, queen J-dive role)
- **Total messages**: ~80+
- **Distinct users**: ~35+ (woodie, Mario [DFW], Prashant, Asher, Kevin J [PDX], Ram, Marshall, Abby [bmore], Blue Chris [SF], DaemonXI [DEN], Don [PDX], Elyse [SLC], Jay Ryan [MPLS], Jeff [SLC], Jess [SF], Le Chiffre, Sky [nyc], Will, coppola [CHI], mel b [pdx], rrenaud, Jason [CHA], JoshB, Felix [SF], lizz, loproto [SEA], Hel [SF], richard [nyc], and more)
- **Key quotes**:
  - "I think we are in a queen-weak era where most warriors are better at killing queens than queens are at resisting strikes" (woodie)
  - "queens are still significantly stronger than warriors... strong queens going something like 30-9 over 5 games" (Asher)
  - "your job as the queen isn't to kill warriors and you go if you have to on your way to a gate" (Jess)
  - "But New York almost entirely succeeds through Facilitating play" (woodie)
- **Data approach**: Compute queen K/D ratio (queen kills warriors vs warriors kill queen) from playerKill events filtering by position 1,6. Track this ratio over time to see if there's a shift. Correlate queen kill efficiency with gate control and win rate. Compare across maps and skill tiers.

### 7. Game Duration by Map and Famine Mechanics

How long do games last per map? What's the longest Day game? How often does famine occur in tournaments?

- **Chunks**: 71-72, 91
- **Conversation bursts**:
  - `3/15/24 -- 4/12/24` (longest Day map games, famine timer history)
  - `6/6/25 -- 7/16/25` (famine frequency at GDC, Coronation finals pain)
- **Total messages**: ~55+
- **Distinct users**: ~20+ (JoshB, Lex, Marshall, Carlitos, Hel [SF], Jess [SF], TheAwfulDin (Ryan), dnb303, EWOK, Le Chiffre, Ram, Jc [CIN], coppola [CHI], Bueno, Felix [SF], tyler [nyc], Ammar, and more)
- **Key quotes**:
  - "KQ26...28 minutes, 21 seconds" (Hel, citing longest pre-famine-timer game)
  - "GDC had far more famines per series than any other tournament" (claim to verify)
- **Data approach**: Compute game duration from gamestart→victory timestamps. Group by map, compute mean/median/percentiles. Count famine frequency (games where all berries deposited). Compare pre/post famine timer eras.

---

### 8. Queen Egg (Maiden) Resource Equivalence

How much is a queen life worth in terms of berries or snail progress? Players attempt to quantify the exchange rate.

- **Chunks**: 100
- **Conversation bursts**:
  - `10/17/25 -- 1/20/26` (queen egg value debate, "1 egg = 4 berries" claim)
- **Total messages**: ~50
- **Distinct users**: 17 (Abby [bmore], Asher, Blue Chris [SF], Clarky, Don [PDX], EliAzizy, Galgo, Jeff [SLC], JoshB, TJ [CHA], Will, alicemyslime [NYC], coppola [CHI], obo (jeff|sea), rrenaud, tyler [nyc], woodie)
- **Key quotes**:
  - "in an average skill 2025 portland scene game, the 1 queen egg would be worth about 4 berries" (Don)
  - "it's 1/3 of the progress for each objective" (response)
- **Data approach**: Use win_probability field from events. Measure the change in win probability when a queen dies (blessMaiden events for egg loss) vs when N berries are deposited. Derive empirical "exchange rate" between queen lives and berry/snail progress.

---

### 9. Berry Economy and Berry Kicking

How effective is berry kicking? Does berry deposit rate correlate with wins? What about multi-person berry runs?

- **Chunks**: 2-3, 28-35, 45, 87-88
- **Conversation bursts**:
  - `8/19/22 12:59 PM -- 8/22/22` (berry kick technique, Monopoli's mapping)
  - `10/18/22 -- 10/19/22` (berry economy, multi-person runs)
  - `1/27/23 -- 1/31/23` (berry kick meta as objective path)
  - `5/7/25 -- 5/27/25` (NY1's Day Berry strategy, hidden objective placement)
- **Total messages**: ~35+
- **Distinct users**: ~20+ (Jess [SF], BrianM [SF], JoshB, Monopoli, rrenaud, Elyse [SLC], Galgo, bigdr00, smokefucius, Blue Chris [SF], Felix [SF], woodie, and more)
- **Key quotes**:
  - "You just gotta go for a lot of berry kicks... Ignore winning, get kick-ins" (Jess [SF])
  - "NY1's 3-1 Day Berry has been super fascinating to look at" (rrenaud)
- **Data approach**: Analyze berryKickIn event frequency by map, team, and location. Correlate berry deposit rate with win probability. Compare games with high kick-in counts vs low. Track which teams rely on kicks vs standard deposits.

---


### 10. Game State Categorization Framework (9 States) and Strategic Volatility

Asher's framework: Neutral, You're Wiped, They're Wiped, etc. Can we classify game events into these states and measure win probability per state? This is novel analysis that hasn't been done before but is very feasible — all the inputs (warrior counts, berry progress, snail position, queen lives) are derivable from event data.

- **Chunks**: 105-106
- **Conversation bursts**:
  - `1/21/26 -- 2/6/26` (game state framework proposal, volatility/positional debt discussion)
- **Total messages**: ~100
- **Distinct users**: ~30+ (Asher, Brian Aston, Clarky, Don [PDX], Elyse [SLC], Jc [CIN], Jeff [SLC], JoshB, Marshall, Monopoli, QweenBilly, Sky [nyc], coppola [CHI], Abby [bmore], Alacrity, Blue Chris [SF], Ram, rrenaud, smokefucius, tyler [nyc], woodie, and more)
- **Data approach**: Build a state classifier from event sequences: warrior counts (from playerKill/spawn events), berry counts (from berryDeposit), snail position (from getOnSnail/snailEat), gate control (from warrior spawn locations), queen lives remaining. Reconstruct game state at every event timestamp. Measure win probability per state and transition probabilities between states. Test if high-volatility games (frequent state changes) favor one strategy over another. This would be a genuinely new contribution to KQ analytics.

### 11. Player Career Progression (Drone → Warrior → Queen)

Do players follow the traditional progression path? How long does it take? Do players who skip steps perform worse?

- **Chunks**: 58-60
- **Conversation bursts**:
  - `4/7/23 -- 7/20/23` (skipping drone→warrior→queen path, queen development)
- **Total messages**: ~5
- **Distinct users**: 4 (lizz, Jess [SF], Prashant, alicemyslime [NYC])
- **Key quote**: "How do people feel about skipping the whole drone→warrior→first speed→and then→Queen process?" (lizz)
- **Data approach**: Track each player's spawn positions (1,6=queen, 2-5/7-10=warrior/drone) across games chronologically using playernames events. Map each player's role distribution over time to see if they follow drone→warrior→queen progression. Compare win rates of players who followed traditional progression vs those who skipped roles.

