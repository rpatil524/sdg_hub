# Flow Discovery

Learn how to discover, organize, and manage flows in SDG Hub. The discovery system automatically finds and registers flows, making them available for use without manual configuration.

## 🔍 Auto-Discovery System

SDG Hub automatically discovers flows in the `src/sdg_hub/flows/` directory using a hierarchical organization system.

### Discovery Process

```python
from sdg_hub.core.flow import FlowRegistry

# Auto-discover all flows in the system
FlowRegistry.discover_flows()

# This scans the flows directory and registers all valid flows
# No manual registration required!
```

### Discovery Locations

The discovery system searches these locations:

```
src/sdg_hub/flows/                    # Built-in flows
├── qa_generation/                    # Question-answer generation
│   ├── document_grounded_qa/
│   │   └── multi_summary_qa/
│   │       └── instructlab/
│   │           ├── flow.yaml
│   │           ├── atomic_facts.yaml
│   │           └── detailed_summary.yaml
│   └── simple_qa/
│       └── flow.yaml
├── text_processing/                  # Text manipulation flows
│   ├── summarization/
│   └── classification/
└── evaluation/                       # Quality assessment flows
    ├── quality_assessment/
    └── bias_detection/

```

## 📋 Flow Registry Operations

### Listing Available Flows

```python
from sdg_hub.core.flow import FlowRegistry

# Discover flows first
FlowRegistry.discover_flows()

# List all available flows
all_flows = FlowRegistry.list_flows()
print(f"Found {len(all_flows)} flows:")
for flow_name in all_flows:
    print(f"  • {flow_name}")

# Output:
# Found 3 flows:
#   • Advanced Document Grounded Question-Answer Generation Flow for Knowledge Tuning
#   • Simple QA Generation Flow
#   • Document Summarization Flow
```

### Getting Flow Information

Access detailed flow metadata and configuration:

```python
from sdg_hub.core.flow import FlowRegistry, Flow

# Get metadata for a specific flow
flow_name = "Advanced Document Grounded Question-Answer Generation Flow for Knowledge Tuning"
metadata = FlowRegistry.get_flow_metadata(flow_name)

if metadata:
    print(f"Flow: {metadata.name}")
    print(f"Version: {metadata.version}")
    print(f"Author: {metadata.author}")
    print(f"Description: {metadata.description}")
    print(f"Tags: {', '.join(metadata.tags)}")
    print(f"Recommended model: {metadata.recommended_models.get('default', 'Not specified')}")

# Load flow and get detailed information
flow_path = FlowRegistry.get_flow_path(flow_name)
flow = Flow.from_yaml(flow_path)

# Get comprehensive flow info
info = flow.get_info()
print(f"Total blocks: {info['total_blocks']}")
print(f"Block sequence: {', '.join(info['block_names'])}")

# Get dataset requirements
requirements = flow.get_dataset_requirements()
if requirements:
    print(f"Required columns: {requirements.required_columns}")
    print(f"Description: {requirements.description}")
    print(f"Min samples: {requirements.min_samples}")

# Get model recommendations
recommendations = flow.get_model_recommendations()
print(f"Default model: {recommendations.get('default')}")
print(f"Compatible models: {recommendations.get('compatible', [])}")
```

### Getting Flow Paths

```python
# Get the file path for a flow
flow_path = FlowRegistry.get_flow_path(flow_name)
print(f"Flow located at: {flow_path}")

# Use the path to load the flow
from sdg_hub.core.flow import Flow
flow = Flow.from_yaml(flow_path)
```

## 🔎 Searching and Filtering Flows

### Search by Tags

```python
# Search for flows with specific tags
qa_flows = FlowRegistry.search_flows(tag="question-generation")
print(f"Q&A Generation flows: {qa_flows}")

educational_flows = FlowRegistry.search_flows(tag="educational")
print(f"Educational flows: {educational_flows}")

document_flows = FlowRegistry.search_flows(tag="document-processing")
print(f"Document processing flows: {document_flows}")
```


## 📊 Flow Organization Patterns

### Hierarchical Organization

Flows are organized in a logical hierarchy:

```
flows/
├── domain/              # By problem domain
│   ├── qa_generation/
│   ├── text_processing/
│   ├── evaluation/
│   └── data_preparation/
├── use_case/            # By specific use case
│   ├── document_grounded_qa/
│   ├── conversational_qa/
│   └── multi_turn_dialogue/
└── variant/             # By implementation variant
    ├── instructlab/     # InstructLab-specific
    ├── simple/          # Simplified version
    └── advanced/        # Feature-rich version
```

### Flow Naming Conventions

Follow consistent naming patterns:

```yaml
# Good naming examples
metadata:
  name: "Advanced Document Grounded Question-Answer Generation Flow for Knowledge Tuning"
  name: "Simple Text Summarization Flow"  
  name: "Multi-Turn Dialogue Generation with Context Tracking"

```

### Directory Structure Guidelines

Organize flows logically:

```
qa_generation/                           # Primary domain
├── document_grounded_qa/                # Specific approach
│   ├── multi_summary_qa/                # Implementation variant
│   │   ├── instructlab/                 # Framework-specific
│   │   │   ├── flow.yaml               # Main flow definition
│   │   │   ├── README.md               # Flow documentation
│   │   │   ├── atomic_facts.yaml       # Supporting templates
│   │   │   ├── detailed_summary.yaml
│   │   │   └── generate_questions_responses.yaml
```

## 🏷️ Flow Categorization and Tagging

### Standard Tag Categories

Use consistent tags for discoverability:

#TODO: Add tag categories


## 🚀 Next Steps

Master flow discovery and organization:

- **[Custom Flows](custom-flows.md)** - Build and organize your own flows
- **[YAML Configuration](yaml-configuration.md)** - Advanced configuration techniques
- **[Development Guide](../development.md)** - Contribute flows to the ecosystem
- **[API Reference](../api-reference.md)** - Complete technical documentation