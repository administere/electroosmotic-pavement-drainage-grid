# EODG: Electroosmotic-Enhanced Composite Drainage Geogrid for Highway Interlayer

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-green)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-theoretical%20validation%20passed-brightgreen)]()
[![Standards](https://img.shields.io/badge/standards-JTG%20D50%20%7C%20D32%20%7C%20F40-orange)]()

**电渗增强复合排水格栅** — A physics-verified, patent-novel design that combines planar interdigitated carbon-fiber electrodes with a 3D HDPE drainage core to actively remove trapped water from highway pavement interlayers.

> ⚡ Electroosmosis releases bound water → 🌊 3D core drains it by gravity → ✅ 3 L/m² in 24h → 🔋 Only 116 Wh/m²/day at 24V DC

---

## The Problem

Highway asphalt pavements trap water at the interface between the asphalt layer and the water-stable base. This interlayer water causes:

- **Frost heave & thaw settlement** in seasonally frozen regions
- **Mud pumping** under repeated traffic loading
- **Delamination** from hydraulic pressure build-up
- **50%+ reduction** in fatigue life (AASHTO mechanistic-empirical studies)

Conventional drainage (porous asphalt, edge drains) struggles with **capillary-bound water** — the thin water films held by surface tension in fine pores.

## The Insight

**Electroosmosis alone is too slow for bulk water transport** (~0.7 mm/h in porous media). Our key realization:

> Don't use electroosmosis to *move* water 20 cm to a drain.  
> Use it to *release* water from capillary bondage.  
> Let gravity do the rest.

This paradigm shift enables a single-layer, 10 mm-thick geogrid that fits within standard asphalt paving.

## Architecture

```
  Asphalt Overlay (≥14 cm)
  ─────────────────────────────────────
  Modified Asphalt Bonding Layer
  ─── Electrode / Drainage Plane ───────  ← Single 10 mm layer
  [Anode+]  [Cathode-]  [Anode+]  [Cathode-]  ← Interdigitated CF tows
  ═══════ 3D HDPE Drainage Core ═══════        ← 8 mm cuspated core
  Nonwoven Geotextile Separator
  ─── Water-Stable Base ───────────────
  ←  d=0.30 m →   period = 0.60 m
```

**Three components integrated in one plane:**
1. **Carbon fiber interdigitated electrodes** — 24V DC, anodes & cathodes alternate at 30 cm pitch
2. **3D HDPE drainage core** — 8 mm thick, 85+ kPa compressive strength, 2×10⁻³ m²/s transmissivity
3. **Silane-treated geotextile** — enhances zeta potential to 50 mV; bonds to asphalt

## Key Design Parameters

| Parameter | Symbol | Value | Unit |
|-----------|:------:|:-----:|------|
| Operating voltage | V | 24 | V DC |
| Electrode pitch | d | 0.30 | m |
| Electrode width | w | 0.015 | m |
| Grid total thickness | — | 10 | mm |
| Asphalt overlay | h | ≥ 0.14 | m |
| Duty cycle | — | 1h ON / 3h OFF | — |
| Interface cohesion | c | 85 | kPa |
| Interface friction angle | φ | 30 | ° |
| Zeta potential (modified) | ζ | 50 | mV |

## Verification Results

All six acceptance criteria pass (see `工程验收报告.md` for full details):

| # | Criterion | Limit | Original | **Optimized** | Status |
|:--:|-----------|:-----:|:--------:|:------------:|:------:|
| C-1 | Current density (avg) | ≤ 1.0 A/m² | 1.27 | **0.20** | ✅ |
| C-2 | 24h drainage | ≥ 3.0 L/m² | 1.05 | **>> 3.0** | ✅ |
| C-3a | Temperature rise (avg) | ≤ 5.0 °C | 6.11 | **0.56** | ✅ |
| C-3b | Peak grating temp | ≤ 50 °C | 41.1 | **37.3** | ✅ |
| C-4 | Shear safety factor | ≥ 1.50 | 1.31 | **1.73** | ✅ |
| C-5 | Daily energy | reference | 1463 Wh | **116 Wh** | ↓92% |

### Verification Figures

**Electric Field FDM (Task 1):**
![E-field Verification](verify_task1.png)

**Drainage Performance (Task 2):**
![Drainage](verify_task2.png)

**Thermal Safety (Task 3):**
![Thermal](verify_task3.png)

**Shear Safety (Task 4):**
![Shear](verify_task4.png)

## Competitive Landscape

| Feature | EKG (UK, commercial) | Chinese Patents | **This Work** |
|---------|:---:|:---:|:---:|
| Electrode layout | Vertical layers (20–40 cm apart) | Vertical layers | **Planar interdigitated** |
| Drainage mechanism | EO-only transport | EO-only | **EO-release + gravity** |
| Operating voltage | 60–100 V | varies | **24 V (SELV)** |
| Installation thickness | Multi-layer, 20+ cm | Multi-layer | **Single-layer, 10 mm** |
| Target application | Slopes, soft soils | Embankments | **Pavement interlayer** |
| Power consumption | Continuous, high | N/A | **Intermittent, 116 Wh/m²/day** |

The **planar interdigitated electrode + 3D core + EO-release paradigm** combination appears novel. No existing product or patent uses this architecture for pavement interlayer drainage.

## Repository Structure

```
├── README.md                          ← This file
├── 导电排水格栅设计方案.md              ← Design specification (Chinese)
├── 物理验证报告.md                      ← Original design verification (Chinese)
├── 工程验收报告.md                      ← Optimized design acceptance (Chinese)
├── .gitignore
│
├── original_design/                    ← Original 48V, d=0.4m, pure EO
│   ├── task1_electric_field.py         ← 2D FDM electric field
│   ├── task2_drainage.py               ← EO drainage rate
│   ├── task3_thermal.py                ← Thermal safety
│   └── task4_shear.py                  ← Interlayer shear
│
├── optimized_design/                   ← Optimized 24V, d=0.3m, EO+gravity
│   ├── verify_task1.py                 ← 2D FDM (sparse direct solver)
│   ├── verify_task2.py                 ← EO + 3D core gravity drainage
│   ├── verify_task3.py                 ← Thermal (steady + transient)
│   └── verify_task4.py                 ← Shear (Boussinesq-Foster&Ahlvin)
│
└── figures/                            ← Generated output figures (*.png)
```

## Quick Start

```bash
# Requirements
pip install numpy scipy matplotlib

# Run optimized design verification
cd optimized_design
python verify_task1.py    # 2D FDM electric field
python verify_task2.py    # Drainage assessment
python verify_task3.py    # Thermal analysis
python verify_task4.py    # Shear safety factor

# Each script prints results to stdout and saves figures + .npz data
```

## Physics Methods

| Task | Method | Equation |
|------|--------|----------|
| Electric Field | Finite Difference Method (FDM) + sparse direct solver | ∇·(σ∇V) = 0 |
| Electroosmosis | Helmholtz-Smoluchowski + porous media correction | v = (εᵣε₀ζ/η)·E·n/τ |
| Gravity Drainage | Darcy's law in 3D core | q = θ · i |
| Thermal | 1D steady-state Fourier + first-order transient | ΔT = P·h/k |
| Shear | Elastic half-space (Boussinesq) + Mohr-Coulomb | FS = (c+σₙtanφ)/τₘₐₓ |

## Standards Compliance (China)

| Standard | Scope | Status |
|----------|-------|:------:|
| JTG D50-2017 | Asphalt pavement design | ✅ Drainage & interlayer requirements met |
| JTG/T D32-2012 | Geosynthetic applications | ✅ Material tests applicable |
| JTG F40-2004 → JTG 3640-20XX | Construction & interlayer functional layer | ⚠️ New "functional layer" chapter (pending) |
| JTG E50-2006 | Geosynthetic testing | ⚠️ No conductive material test method (gap) |
| GB/T 16895 / GB 50054 | Low-voltage electrical | ✅ 24V SELV — intrinsically safe |
| MOT "Quality Project" 2024 | New tech demonstration pathway | ⚠️ Requires 1+ year trial section |

> **Regulatory path**: The product must enter through the MOT's "Four-New Technology" (新技术/新工艺/新材料/新设备) demonstration project channel. Industry standards for conductive geosynthetics do not yet exist in China — this is both a gap and an opportunity.

## Limitations & Disclaimer

⚠️ **This is a theoretical physics validation.** The results are based on simplified models:

1. **Electroosmosis**: Helmholtz-Smoluchowski assumes ideal capillary bundles. Real pavement pore geometry, water chemistry (pH, ionic strength), and temperature effects will alter zeta potential significantly.
2. **Thermal**: 1D steady-state neglects lateral heat spreading, solar radiation, wind cooling, and asphalt viscoelastic heating.
3. **Shear**: Elastic half-space ignores asphalt viscoelasticity, dynamic loading, and imperfect interlayer bonding.
4. **Durability**: Carbon fiber anode corrosion, geotextile clogging, and HDPE creep are modeled with conservative reduction factors but need long-term experimental validation.

**Full-scale accelerated pavement testing (APT) and field trial sections are required before engineering deployment.**

## Citation

If you use this work, please cite:

```bibtex
@misc{eodg2026,
  title     = {EODG: Electroosmotic-Enhanced Composite Drainage Geogrid for Highway Interlayer},
  author    = {Wayne},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/wayne-abc/eodg}
}
```

## License

MIT License — see [LICENSE](LICENSE) file.

---

*Designed by a computational physics engineer. Verified with Python + NumPy + SciPy + Matplotlib. July 2026.*
