.PHONY: install pipeline test app
install: ; pip install -r requirements.txt -r requirements-dev.txt
pipeline: ; python run_pipeline.py
test: ; python -m pytest -q tests
app: ; streamlit run app.py
