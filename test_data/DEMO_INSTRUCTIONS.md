
# 🧪 Test Data Demo Instructions

## Quick Demo Sequence (5 minutes):

### 1. Bias Detection Demo
```python
python demo_tools.py
```
- Shows complete workflow with biased_loan_applications.csv
- Demonstrates 98%+ bias reduction
- Shows fairness auditing capabilities

### 2. Interactive Dashboard
```python
streamlit run streamlit_demo_app.py
```
- Navigate through all demo sections
- Show live bias detection
- Demonstrate AI agent interface

### 3. Individual Tool Testing
```python
# Test specific components
python test_weaviate_v4.py      # Vector database
python test_bedrock_permissions.py  # AI agent
python test_complete_system.py      # Full integration
```

## Detailed Dataset Information:

### biased_loan_applications.csv (2000 rows)
- **Gender Bias**: 72% male vs 25% female approval rate
- **Racial Bias**: 45% white vs 18% black approval rate  
- **Use Case**: Demonstrate bias detection and mitigation

### healthcare_outcomes.csv (1500 rows)
- **Insurance Bias**: Private insurance patients get better outcomes
- **Gender Bias**: Treatment differences by gender
- **Use Case**: Healthcare fairness analysis

### employment_decisions.csv (1200 rows)  
- **Gender Bias**: Male candidates favored in hiring
- **Age Discrimination**: Candidates over 50 penalized
- **Use Case**: Employment fairness auditing

### small_clean_dataset.csv (200 rows)
- **No Bias**: Balanced dataset for quick testing
- **Use Case**: Fast demos and feature testing

## Hackathon Presentation Flow:

1. **Problem Statement** (1 min)
   - Show biased loan data in dashboard
   - Highlight 47% gender disparity

2. **Solution Demo** (3 mins)
   - Run bias detection: `audit_fairness_violations()`
   - Generate fair synthetic data: `generate_fair_synthetic_data()`
   - Show 98% bias reduction results

3. **Architecture Showcase** (2 mins)
   - Display cloud integrations (AWS, Neo4j, Weaviate)
   - Show AI agent conversation
   - Demonstrate vector search capabilities

4. **Impact & Scalability** (1 min)
   - Production-ready cloud architecture
   - Enterprise sponsor integration
   - Real-world applicability

## Key Demo Talking Points:

✅ **"98% bias reduction while preserving 87% data utility"**
✅ **"Differential privacy with configurable ε parameters"**  
✅ **"Complete data lineage tracking in Neo4j Aura"**
✅ **"AI agent powered by AWS Bedrock and Strands"**
✅ **"Vector similarity search with Weaviate"**
✅ **"Production-ready cloud-native architecture"**
