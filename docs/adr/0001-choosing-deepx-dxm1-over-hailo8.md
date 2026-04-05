# ADR-0001: Hardware Selection — DeepX DX-M1 vs. Hailo-8

This Architectural Decision Record (ADR) outlines the technical rationale for selecting the **DeepX DX-M1** AI Accelerator over the **Hailo-8 (HM218B1C2FA)** for the **LattePanda 3 Delta** single-board computer (SBC).

---

## 1. Context & Constraints
The **LattePanda 3 Delta (Intel N5105)** is a compact but resource-constrained SBC. To enable high-performance AI vision (25+ TOPS), we must navigate significant hardware limitations:
* **PCIe Topology:** The Intel Jasper Lake platform has only 8 total PCIe lanes distributed across multiple Root Ports. By default, Root Ports are set to "Auto" speed negotiation, which can cause DMA initialization failures for the DX-M1 firmware at Gen1 (2.5 GT/s). The fix is to force Root Ports 1, 5, and 6 to Gen2 explicitly in BIOS — no SATA controller changes are required.
* **Thermal Environment:** The project uses the **Titan Case** (ABS plastic), which traps heat more than aluminum enclosures.

---

## 2. Decision: DeepX DX-M1
The **DeepX DX-M1** was selected as the optimal "balanced" partner for this specific SBC.

### 2.1 Storage Strategy: Dual M.2 Slots
Both the M-Key (DX-M1) and B-Key (SATA SSD) slots operate simultaneously without modification:
* **M-Key slot:** DeepX DX-M1 AI Accelerator (PCIe Gen2 x2).
* **B-Key slot:** Transcend 430S SATA SSD — primary OS and data storage.
* **64GB eMMC:** Reserved as a recovery OS image; auto-mount disabled via udev rule.

> **Note:** Early documentation referenced a "Hybrid-External" USB enclosure boot strategy based on a mistaken assumption that the SATA controller had to be disabled for the DX-M1. This was incorrect. Both slots coexist when PCIe Root Ports 1, 5, and 6 are forced to Gen2. No external enclosure is required.

### 2.2 Memory & Host Relief
* **DX-M1 Advantage:** Features **4GB of dedicated LPDDR5** memory. 
* **Impact:** On an 8GB RAM system, the DX-M1 stores model weights internally. The **Hailo-8** lacks dedicated large-scale memory and must "stream" weights from the host RAM, which would starve the Ubuntu 24.04 OS.

### 2.3 Thermal & Power Stability
* **Efficiency:** The DX-M1 operates between **2W – 5W**. In the plastic **Titan Case**, it maintains a stable temperature (approx. **61°C**), whereas the **Hailo-8** (up to 8.2W) risks thermal throttling.
* **Signal Integrity:** The DX-M1 is validated to run reliably at **PCIe Gen2 x2** speeds on this board, avoiding the DMA header overflows found at Gen1 speeds.

---

## 3. Comparison Summary

| Feature | DeepX DX-M1 | Hailo-8 |
| --- | --- | --- |
| Power Profile	| 2W - 5W (Stable) | Up to 8.2W (Aggressive) |
| Onboard Memory | 4GB LPDDR5 (Host Relief) | Internal SRAM Only |
| Thermal Fit | Ideal for Plastic Case | Requires Metal Heatsink |
| Storage Impact | Requires External USB Boot | High Host RAM Overhead |

---

## 4. Consequences
* **BIOS Requirement:** Must set PCIe Root Ports 1, 5, and 6 to **Gen2**. SATA Controller remains **Enabled**.
* **Storage:** Both M-Key (DX-M1) and B-Key (SATA SSD) slots work simultaneously — no external enclosures required.
* **Software:** Utilizes the **DXNN SDK** and **DX-RT** runtime.

**Status:** Accepted and Validated.

---
