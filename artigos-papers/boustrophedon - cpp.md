# Boustrophedon / Lawnmower Coverage Path Planning

Papers about back-and-forth coverage patterns and their use in robotics and UAVs. Links checked on September 25, 2026.

The Tello demo in `scripts/zigzag_search.py` uses a predefined lawnmower pattern. Full boustrophedon cellular decomposition additionally divides an environment into cells and plans coverage between them, as described in the foundational paper below.

| Paper | Year | Relevance and link |
| --- | --- | --- |
| **Coverage Path Planning: The Boustrophedon Cellular Decomposition** — Howie Choset and Philippe Pignon | 1997 conference paper | Foundational method: divide the environment into cells, cover each with back-and-forth motion, and connect the cells. [Full paper, CMU PDF](https://www.ri.cmu.edu/pub_files/pub4/choset_howie_1997_3/choset_howie_1997_3.pdf). [CMU publication record](https://publications.ri.cmu.edu/coverage-path-planning-the-boustrophedon-decomposition). |
| **Survey on Coverage Path Planning with Unmanned Aerial Vehicles** — Tauã M. Cabreira, Lisane B. Brisolara, and Paulo R. Ferreira Jr. | 2019 | Overview of UAV coverage methods, including back-and-forth patterns and cellular decomposition. A useful starting point for drone applications. [Open-access article](https://www.mdpi.com/2504-446X/3/1/4). |
| **Revisiting Boustrophedon Coverage Path Planning as a Generalized Traveling Salesman Problem** — Rik Bähnemann et al. | 2019, FSR; linked preprint | Optimizes boustrophedon sweeps for rotary-wing aerial vehicles, handles obstacles between cells, and includes validation on a real aerial platform. [Paper and PDF access](https://arxiv.org/abs/1907.09224). |
| **Complete coverage path planning for aerial vehicle flocks deployed in outdoor environments** | 2019 | Uses back-and-forth coverage within subregions assigned to multiple UAVs, accounting for no-fly zones. [Publisher page and abstract](https://www.sciencedirect.com/science/article/abs/pii/S0045790618319426). [DOI](https://doi.org/10.1016/j.compeleceng.2019.02.024). |
| **UAV survey coverage path planning of complex regions containing exclusion zones** — Shadman Tajwar Shahid, Shah Md. Ahasan Siddique, and Md. Mahidul Alam | 2024, arXiv preprint | Generates continuous boustrophedon paths for complex survey regions with exclusion zones using polygon decomposition and merging. [Paper and PDF access](https://arxiv.org/abs/2411.07053). |

Suggested reading order: start with the UAV survey, then the foundational Choset–Pignon paper, followed by Bähnemann et al. for an aerial implementation with obstacles.
