# \# PlantVision AI

# 

# \*\*From Plant Images to Agricultural Intelligence.\*\*

# 

# > ✅ Current status: \*\*Phase 2 (plant classifier) is IMPLEMENTED and evaluated.\*\*

# > The classifier is trained on PlantVillage and deployed in a working Streamlit

# > preview app. Remaining modules (disease-only head, pest detection, severity,

# > health score, RAG recommendations) are PLANNED — see `docs/ROADMAP.md`.

# 

# \## Overview

# 

# PlantVision AI is a modular, AI-powered plant health analysis platform. It is

# designed to take a single plant image and extract the maximum amount of

# scientifically defensible information about that plant's health — plant

# identification, disease indicators, visible symptoms, pests (where a model

# supports it), estimated severity, a transparent health score, and

# knowledge-grounded recommendations.

# 

# \## Problem

# 

# Small farmers, students, and plant owners often lack fast, low-friction access

# to a first-pass assessment of plant health. Professional diagnosis is

# valuable but not always immediately available.

# 

# \## Solution

# 

# An image-in, report-out decision-support tool that is explicit about what it

# can and cannot determine from a photo alone (see `docs/LIMITATIONS.md`), and

# that never presents unvalidated or fabricated results as real.

# 

# \## Architecture

# 

# See `docs/ARCHITECTURE.md` for the full module breakdown and data flow.

# 

# \## Feature Status

# 

# | Feature | Status |

# |---|---|

# | Project scaffold, config, docs | \*\*IMPLEMENTED\*\* |

# | Dataset pipeline (PlantVillage) | \*\*IMPLEMENTED\*\* |

# | Plant identification | \*\*IMPLEMENTED\*\* (94.1% test accuracy) |

# | Disease detection (via plant label) | \*\*IMPLEMENTED\*\* (part of classifier output) |

# | Streamlit preview app | \*\*IMPLEMENTED\*\* (Green Field UI) |

# | Symptom analysis | PLANNED |

# | Pest detection | PLANNED |

# | Severity estimation | PLANNED |

# | Health score engine | PLANNED (config exists, engine does not) |

# | Explainability (Grad-CAM) | PLANNED |

# | Knowledge base | PLANNED |

# | RAG assistant | PLANNED |

# | Recommendation engine | PLANNED |

# | Plant history (SQLite) | PLANNED |

# | Early warning / environmental integration | FUTURE EXTENSION |

# 

# \## Technology Stack (implemented)

# 

# \- Python 3.11

# \- PyTorch / TorchVision (MobileNetV2 transfer learning)

# \- OpenCV (auto-crop preprocessing)

# \- Streamlit (application UI)

# \- scikit-learn (evaluation metrics)

# \- matplotlib / seaborn (charts)

# 

# Planned additions:

# \- Ultralytics YOLO (pest detection)

# \- SQLite (plant history)

# \- RAG stack (vector store + LLM) — Phase 9

# 

# \## Dataset

# 

# \*\*PlantVillage\*\* — 38 classes across 14 crops (color images).

# Train/validation/test split produced by `scripts/split\_dataset.py`.

# See `docs/DATASET\_PLAN.md` and `reports/dataset\_inspection\_summary.md`.

# 

# \## Installation

# 

# ```bash

# python -m venv venv

# \# Windows: venv\\Scripts\\activate

# \# Linux/Mac: source venv/bin/activate

# pip install -r requirements.txt

# cp .env.example .env

