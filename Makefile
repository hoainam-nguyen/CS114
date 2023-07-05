run_app:
	python app.py

run_web:
	streamlit run ui.py --server.address 0.0.0.0 --server.port 7778
