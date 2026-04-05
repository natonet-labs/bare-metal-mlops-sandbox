# The "Level-Fill" Thermal Sandwich Guide
**Optimized for: DeepX DX-M1 + LattePanda 3 Delta + Titan Case**

In a small plastic case like the **Titan Case**, AI modules like the **DeepX DX-M1** can get very hot (60°C+) because they are tucked away from the main fans. This guide shows you how to build a "Thermal Sandwich" to move that heat into a heatsink and out of the case.

### 1. The Strategy: "Flat is Better"
* **The Problem:** The AI board is uneven, with a high (the main chip) and low "valleys" (the circuit board around it).
* **The Solution:** Use different thicknesses of thermal pads to make the board perfectly flat before adding a heatsink.
* **The Result:** Stable idle temperatures of **52°C** (down from 60°C+ original estimates).

### 2. What You Need
* **1mm Thermal Pad:** To fill the "valleys" (low spots).
* **0.5mm Thermal Pad:** For the final "cap" (the flat bridge).
* **Small Aluminum Heatsink:** To soak up the heat.
* **Kapton Tape:** For electrical safety.

---

### 3. The Build Steps

#### Step A: Safety First (The Tape)
Cover the tiny gold components around the main NPU chip with **Kapton tape**. This creates a safety barrier so the thermal pads don't cause any electrical shorts on the board.

#### Step B: The "Valley-Fill" (1mm Pads)
Do not just slap a thick pad over everything. This creates air pockets that act like a heat blanket.
1.  Cut your **1mm pad** into small pieces.
2.  Place them into the **recesses (low spots)** around the main chip.
3.  **Goal:** Bring the level of the "valleys" up until they are the same height as the main chip.

#### Step C: The "Cap" (0.5mm Strip)
Lay one long, continuous **0.5mm strip** across the entire module. This strip should cover both your 1mm filler pieces AND the main chip.
* This creates a perfectly flat, solid "bridge" for your heatsink to sit on.

#### Step D: The "Heatsink & Air Gap"
Place your heatsink on top of the pads. Use rubber bands or case tension to hold it firm.
* **CRITICAL:** Ensure there is a **small gap** between the bottom of the heatsink and the plastic floor of the case.
* **Why?** If the metal touches the plastic, the heat gets trapped. The air gap lets the case fans blow the heat away.

---

### 4. Why This Works
1.  **Zero Flex:** Filling the valleys ensures the board won't "bow" or bend under the weight of the heatsink, protecting the delicate solder joints.
2.  **No Trapped Air:** You've replaced "dead air" (an insulator) with "thermal pads" (conductors).
3.  **Convection:** The air gap at the bottom allows the heatsink to stay cool by letting air pass over its surface.

---
