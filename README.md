# LLM-driven Personalized Garment Synthesis Enhanced by Neuro-Symbolic RAG and Physics-aware Pattern Optimization

This repository implements core components of an **unpublished** integrated framework for AI-driven, personalized garment synthesis. The pipeline turns multi-modal user input (portrait + intent) into high-fidelity garment renders through four collaborative stages; 





## Project structure

Requirement_Analyst: RAGFlow-based intent parsing and ShapeBoost integration scripts.

Generative_Designer: Fine-tuned LDM for PBR textures and DressCode-compatible pattern generators.

Pattern_Adapter: Implementation of the Approximated Adam algorithm and Marvelous Designer API wrappers.

Evaluator: Human-in-the-loop interface for real-time visual assessment and dual-modal feedback.




## Requirements

- **Marvelous Designer** or **CLO** with scripting support (Python host).
- Python 3.9+ (as provided by the vendor host).
- the optimization package uses only the standard library and vendor APIs.
This work integrates several pioneering technologies. Please ensure proper attribution when using this framework:


Garment Generation: DressCode（https://github.com/IHe-KaiI/DressCode）.


Body Reconstruction: Shapeboost（https://github.com/biansy000/ShapeBoost）.


RAG Workflow: RAGFlow（https://ragflow.io/）.



## Setup

1. Clone the repository.
2. Clone all the pioneering technologies
3. Use shapeboost and the code in requirement_analyst to du the biometric extraction, and then analyse it in RAGflow. Generate the patterns using Dresscode; in MD, use API to do the pattern adjustment.




## Citation

The pipeline (Requirement Analyst → Generative Designer → Physics-Driven Deformer → Design Evaluator) and the optimization formulation correspond to an **unpublished** paper. A citation will be added upon publication.



## License

See the repository license file (if present). This project is for research and development use with Marvelous Designer / CLO.



## Contact
If you'd like to experience this framework, feel free to contact me at [hazelzhang112@outlook.com] with your interest. I can share a version of it that I have personally built and set up for further testing and exploration.