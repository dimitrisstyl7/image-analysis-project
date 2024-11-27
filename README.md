# Content-Based Image Retrieval System

## [University of Piraeus](https://www.unipi.gr/en/home/) | [Department of Informatics](https://cs.unipi.gr/en/)
**BSc course**: Image Analysis

**Semester**: 7

**Project Completion Year**: 2024

## Description
This project implements a Content-Based Image Retrieval (CBIR) system using hypergraph-based algorithm. The system aims to retrieve images from a database based on their visual content rather than metadata. By leveraging advanced techniques such as hypergraph construction and similarity calculations, the project enhances the accuracy and efficiency of image retrieval processes.

## Features
- **Content-Based Image Retrieval**: Retrieval of images based on their content using graph-theoretic algorithms.
- **Rank Normalization**: Normalization of the ranking list of images for improved accuracy in retrieval.
- **Hypergraph Construction**: Construction of a hypergraph to represent the relationships between images and their features.
- **Hyperedge Similarities Calculation**: Calculation of similarities between hyperedges of the hypergraph to evaluate the relationship between images.
- **Cartesian Product of Hyperedge Elements**: Calculation of the Cartesian product of hyperedge elements to enhance similarity analysis.
- **Hypergraph-Based Similarity**: Similarity computation based on the constructed hypergraph, allowing for accurate retrieval of relevant images.
- **Pre-trained Neural Network Features**: Extraction of objective features from pre-trained neural networks (such as SqueezeNet, GoogleNet, ResNet) using intermediate hidden layers.
- **Target Image Identification**: Ability to designate specific images as target images and present related images from the database.
- **Accuracy Measurement**: Systematic process for measuring the accuracy of the algorithm and presenting the results.

## How to Run
1. **Clone the repository**:
```bash
git clone https://github.com/dimitrisstyl7/image-analysis-project.git
```
2. **Navigate to the project directory**:
```bash
cd image-analysis-project/Project
```
3. **Create and activate a virtual environment**:

_On Linux/Mac_
```bash
python3 -m venv venv
source venv/bin/activate
```

_On Windows_
```bash
python -m venv venv
venv\Scripts\activate
```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```
5. **Run the program**:

_On Linux/Mac_
```bash
python3 main.py
```
_On Windows_
```bash
python main.py
```

## Contributors
<table>
  <tr>
    <td align="center"><a href="https://github.com/dimitrisstyl7"><img src="https://avatars.githubusercontent.com/u/75742419?v=4" width="100px;" alt="Dimitris Stylianou"/><br /><sub><b>Dimitris Stylianou</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/panagiota02"><img src="https://avatars.githubusercontent.com/u/79789822?v=4" width="100px;" alt="Panagiota Nicolaou"/><br /><sub><b>Panagiota Nicolaou</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/Nikidmp"><img src="https://avatars.githubusercontent.com/u/79726656?v=4" width="100px;" alt="Niki Dimopoulou"/><br /><sub><b>Niki Dimopoulou</b></sub></a><br /></td>
  </tr>
</table>

## Acknowledgments
This project was developed as part of the "Image Analysis" BSc course at the University of Piraeus. Contributions and feedback are always welcome!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
