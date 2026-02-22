VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
FRONTEND_DIR = frontend
BACKEND_DIR = backend

.PHONY: setup
setup: venv install-backend install-frontend build-frontend
	@echo ""
	@echo "Setup complete!"
	@echo "   Run 'make run' to start the application."

.PHONY: venv
venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV_DIR); \
		echo "   Virtual environment created at $(VENV_DIR)/"; \
	else \
		echo "Virtual environment already exists."; \
	fi

.PHONY: install-backend
install-backend: venv
	@echo "Installing Python dependencies..."
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt -q
	@echo "   Backend dependencies installed."

.PHONY: install-frontend
install-frontend:
	@echo "Installing frontend dependencies..."
	cd $(FRONTEND_DIR) && npm install
	@echo "   Frontend dependencies installed."

.PHONY: build-frontend
build-frontend:
	@echo "Building frontend..."
	cd $(FRONTEND_DIR) && npm run build
	@echo "   Frontend built successfully."

.PHONY: run
run: build-frontend
	@echo "Starting application..."
	@echo "   Backend:  http://localhost:8000"
	@echo "   Frontend: http://localhost:5173 (dev server)"
	@echo "   Press Ctrl+C to stop both."
	@echo ""
	@trap 'kill 0' INT; \
		$(PYTHON) $(BACKEND_DIR)/app.py & \
		cd $(FRONTEND_DIR) && npm run dev & \
		wait

.PHONY: run-backend
run-backend:
	@echo "Starting backend at http://localhost:8000"
	$(PYTHON) $(BACKEND_DIR)/app.py

.PHONY: run-frontend
run-frontend:
	@echo " Starting frontend dev server..."
	cd $(FRONTEND_DIR) && npm run dev

.PHONY: clean
clean:
	@echo "Cleaning..."
	rm -rf $(VENV_DIR)
	rm -rf $(FRONTEND_DIR)/node_modules
	rm -rf $(FRONTEND_DIR)/dist
	@echo "   Cleaned."
