# Quantum Forge Engine

![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A revolutionary quantum-inspired system for generating synthetic data that preserves the statistical properties of original datasets while ensuring privacy compliance through advanced differential privacy techniques.

## Overview

Quantum Forge Engine is an enterprise-ready tool built on cutting-edge matrix transformation algorithms, enhanced with an intuitive interface and advanced analytics capabilities. It enables data scientists, researchers, and privacy professionals to create high-quality synthetic datasets for testing, development, and sharing without exposing sensitive information.


## Key Features

### Quantum Data Forge Capabilities
- **Dual Forge Modes**:
  - **Autonomous Mode**: For maximum privacy and simple data structures
  - **Relational Mode**: For preserving complex relationships between variables using advanced Bayesian networks

- **Privacy Protection**:
  - Configurable differential privacy with epsilon parameter control
  - Automatic detection and handling of sensitive attributes
  - Privacy-utility tradeoff visualization and optimization

### Advanced Analytics
- **Comprehensive Data Comparison**:
  - Statistical fidelity metrics between source and quantum data
  - Distribution comparison with multiple visualization options
  - Correlation preservation analysis

- **Visualization Suite**:
  - Interactive histograms, box plots, and scatter plots
  - Correlation heatmaps and mutual information analysis
  - Pair plots for multivariate relationship examination

### Enterprise Features
- **Reporting and Documentation**:
  - Exportable HTML analysis reports
  - Data quality metrics and validation
  - Forge configuration documentation

- **User Experience**:
  - Intuitive web interface with responsive design
  - Progress tracking for long-running operations
  - Comprehensive error handling and logging

## Installation

### Using Poetry (Recommended)
```bash
# Clone the repository
git clone git@github.com:jasmadata/jasma-data-synthesizer-p1-v1.git
cd jasma-data-synthesizer-p1-v1

# Install with Poetry
poetry install

# Run the application
poetry run jasma-data-synthesizer-p1-v1
```

### Using Pip
```bash
# Clone the repository
git clone git@github.com:jasmadata/jasma-data-synthesizer-p1-v1.git
cd jasma-data-synthesizer-p1-v1

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run quantum_forge_engine.py
```

## Quick Start Guide

1. **Launch the Application**: Start the application using one of the methods above
2. **Upload Your Dataset**: Use the file uploader to import your CSV data
3. **Configure Forge Parameters**:
   - Select forge mode (Autonomous or Relational)
   - Set privacy budget (epsilon)
   - Configure number of records to generate
4. **Review and Adjust Attribute Settings**:
   - Verify categorical attribute detection
   - Specify unique identifiers
5. **Generate Quantum Data**: Click the "Forge Quantum Data" button
6. **Analyze and Export**:
   - Explore the quantum data through the visualization tabs
   - Compare with the source dataset
   - Download the quantum data and/or analysis report

## Forge Modes Explained

### Autonomous Mode
This mode synthesizes each attribute independently, preserving the distribution of values within each attribute but not the relationships between attributes.

**Best for**:
- Maximum privacy protection
- Simple data structures
- Scenarios where attribute relationships are not critical
- Faster synthesis of large datasets

### Relational Mode
This mode uses advanced Bayesian networks to model and preserve relationships between attributes, creating more realistic quantum data.

**Best for**:
- Maintaining complex data relationships
- Machine learning training datasets
- When data utility is a priority
- Scenarios requiring realistic data patterns

## Privacy Controls

The epsilon parameter is central to the differential privacy implementation:

| Epsilon Value | Privacy Level | Data Utility | Use Case |
|---------------|--------------|-------------|----------|
| 0.1 - 0.5     | Very High    | Lower       | Highly sensitive data (healthcare, financial) |
| 0.5 - 1.0     | High         | Moderate    | Personal data with some sensitive attributes |
| 1.0 - 3.0     | Moderate     | Good        | General business data |
| 3.0 - 10.0    | Lower        | Excellent   | Less sensitive data, prioritizing utility |

## Technical Requirements

- **Python**: 3.11 or higher
- **Core Dependencies**:
  - Streamlit 1.30.0+
  - Pandas 2.2.0+
  - NumPy 1.24.0+
  - Matplotlib 3.7.0+
  - Seaborn 0.12.0+
  - Scikit-learn 1.2.0+
  - Plotly 5.13.0+
  - SciPy 1.10.0+

## Development

### Setting Up Development Environment
```bash
# Install development dependencies
poetry install --with dev

# Set up pre-commit hooks
pre-commit install
```

### Running Tests
```bash
poetry run pytest
```

### Code Quality Checks
```bash
# Run linting
poetry run flake8

# Run type checking
poetry run mypy .

# Format code
poetry run black .
poetry run isort .
```

## Contributing

We welcome contributions to improve Quantum Forge Engine! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please ensure your code adheres to our quality standards by running the provided linting and testing tools.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Bayesian Networks](https://en.wikipedia.org/wiki/Bayesian_network) - The mathematical foundation for our relational mode
- [Streamlit](https://streamlit.io/) - The framework used for building the web application
- [Differential Privacy](https://en.wikipedia.org/wiki/Differential_privacy) - The privacy model implemented in this tool

---

<p align="center">
  <b>Jasma Team</b><br>
  <i>Advancing Privacy-Preserving Data Science</i><br>
  <a href="jasma.xyz">jasma.xyz</a>
</p>
