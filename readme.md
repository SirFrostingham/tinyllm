# Summary 

This is a project for me to experiment with smaller LLMs to run on very low memory & CPU systems.

# Project Setup Guide

This guide will walk you through setting up a Python virtual environment and installing the required dependencies for your project.

## Prerequisites

Before you begin, ensure that Python is installed on your system. You can download Python from the [official Python website](https://www.python.org/downloads/). During the installation, make sure to select the option to **Add Python to PATH**.

## Setting Up the Virtual Environment

Follow these steps to set up a virtual environment for your project:

### Step 1: Create the Virtual Environment

Navigate to your project directory in the Command Prompt and run the following command to create a virtual environment named `llm-env`:

```bash
cd path\to\your\project  # Navigate to your project directory
python -m venv llm-env   # Create a virtual environment
```

### Step 2: Activate the Virtual Environment

```bash
llm-env\Scripts\activate  # Activate the virtual environment
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt  # Install dependencies from requirements.txt
```

### Step 4: Verify Installation

```bash
pip list  # List installed packages
```

### Deactivating the Virtual Environment

```bash
deactivate  # Deactivate the virtual environment
```

## Summary
Using a virtual environment for Python projects is best practice as it manages project-specific dependencies independently of global Python settings. This setup ensures that your project environment is reproducible on any other machine.

