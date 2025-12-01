# Makefile for dapr-state-cache
# Seguindo regras Clean Code + TDD

.DEFAULT_GOAL := help
.PHONY: help lint test-quick test-coverage clean install dev-install \
	format lint-check lint-fix radon vulture type-check static-analysis security \
	quality metrics health check fix validate wily-init wily-build wily-report

# Variables
UV := uv
PYTHON := python
SRC_PATH := src/dapr_state_cache
TEST_PATH := tests
COVERAGE_THRESHOLD := 100
TARGET ?= $(SRC_PATH) $(TEST_PATH)

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install the package
	$(UV) pip install -e .

dev-install: ## Install with development dependencies
	$(UV) pip install -e ".[dev]"

lint: ## Run linting checks (zero errors required)
	@echo "Running linting checks..."
	$(UV) run ruff check $(SRC_PATH) $(TEST_PATH)
	$(UV) run ruff format --check $(SRC_PATH) $(TEST_PATH)
	@echo "✅ Linting passed - zero errors"

lint-fix: ## Fix linting issues automatically
	@echo "Fixing linting issues..."
	$(UV) run ruff check --fix $(SRC_PATH) $(TEST_PATH)
	$(UV) run ruff format $(SRC_PATH) $(TEST_PATH)

# Individual Quality Tools
format: ## Format code automatically
	@echo "Formatting code..."
	$(UV) run ruff format $(TARGET)
	@echo "✅ Code formatted"

lint-check: ## Check linting without fixes
	@echo "Checking code quality..."
	$(UV) run ruff check $(TARGET)
	@echo "✅ Linting check passed"

radon: ## Analyze code complexity and metrics
	@echo "Analyzing code complexity..."
	$(UV) run radon cc $(TARGET) -s -a
	$(UV) run radon mi $(TARGET) -s
	$(UV) run radon raw $(TARGET) -s
	@echo "✅ Code analysis completed"

vulture: ## Detect unused code
	@echo "Detecting unused code..."
	$(UV) run vulture $(TARGET)
	@echo "✅ Unused code analysis completed"

# Conceptual Commands (Stable Abstractions)
type-check: ## Static type verification
	@echo "Executando verificação de tipos..."
	$(UV) run pyright $(TARGET)
	@echo "✅ Verificação de tipos concluída"

static-analysis: ## Complete static code analysis
	@echo "Executando análise estática completa..."
	$(MAKE) type-check TARGET=$(TARGET)
	$(MAKE) security TARGET=$(TARGET)
	@echo "✅ Análise estática completa"

security: ## Security analysis via Ruff (bandit rules)
	@echo "Running security analysis..."
	$(UV) run ruff check --select=S $(TARGET)
	@echo "✅ Security analysis completed"

# Wily - Maintainability Tracking
wily-init: ## Initialize Wily for maintainability tracking
	@echo "Initializing Wily..."
	printf "y\n10\n$(SRC_PATH)\n" | $(UV) run wily setup
	@echo "✅ Wily initialized"

wily-build: ## Update Wily database with current revision
	@echo "Building Wily database..."
	$(UV) run wily build $(SRC_PATH)
	@echo "✅ Wily database updated"

wily-report: ## Generate maintainability report
	@echo "Generating maintainability report..."
	$(UV) run wily report $(SRC_PATH)
	@echo "✅ Maintainability report generated"

# Combined Quality Commands
quality: ## Complete quality analysis (radon + vulture + type-check + security)
	@echo "Running complete quality analysis..."
	$(MAKE) radon TARGET=$(TARGET)
	$(MAKE) vulture TARGET=$(TARGET)
	$(MAKE) type-check TARGET=$(TARGET)
	$(MAKE) security TARGET=$(TARGET)
	@echo "✅ Complete quality analysis finished"

metrics: ## Code metrics analysis (radon + complexity)
	@echo "Running metrics analysis..."
	$(MAKE) radon TARGET=$(TARGET)
	@echo "✅ Metrics analysis completed"

health: ## Project health check (quality + wily + coverage)
	@echo "Running project health check..."
	$(MAKE) quality
	$(MAKE) wily-build
	$(MAKE) wily-report
	$(MAKE) test-coverage
	@echo "✅ Project health check completed"

# Development Workflows
check: ## Quick development check (lint + types + format-check)
	@echo "Running quick development check..."
	$(UV) run ruff check $(TARGET)
	$(UV) run ruff format --check $(TARGET)
	$(MAKE) type-check TARGET=$(TARGET)
	@echo "✅ Quick check passed"

fix: ## Auto-fix issues (format + lint-fix)
	@echo "Auto-fixing code issues..."
	$(UV) run ruff format $(TARGET)
	$(UV) run ruff check --fix $(TARGET)
	@echo "✅ Code issues fixed"

validate: ## Complete validation before commit
	@echo "Running complete validation..."
	$(MAKE) check TARGET=$(TARGET)
	$(MAKE) quality TARGET=$(TARGET)
	$(MAKE) test-coverage
	@echo "✅ Complete validation passed"

test-quick: lint ## Run linting + unit tests (TDD validation)
	@echo "Running quick test suite (linting + unit tests)..."
	$(UV) run pytest $(TEST_PATH)/unit -v --tb=short
	@echo "✅ Quick tests passed"

test-coverage: ## Run tests with 100% coverage requirement
	@echo "Running tests with coverage analysis..."
	$(UV) run pytest $(TEST_PATH) --cov=$(SRC_PATH) --cov-report=term-missing --cov-report=html --cov-fail-under=$(COVERAGE_THRESHOLD) --cov-branch
	@echo "✅ 100% coverage achieved"

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	$(UV) run pytest $(TEST_PATH)/integration -v

test-all: ## Run all tests (unit + integration + coverage)
	@echo "Running complete test suite..."
	$(UV) run pytest $(TEST_PATH) -v --cov=$(SRC_PATH) --cov-report=term-missing --cov-report=html --cov-fail-under=$(COVERAGE_THRESHOLD) --cov-branch

clean: ## Clean build artifacts and cache
	@echo "Cleaning build artifacts..."
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -not -path "./.venv/*" -delete
	find . -name "__pycache__" -not -path "./.venv/*" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean completed"

build: lint test-coverage ## Build package (after validation)
	@echo "Building package..."
	$(UV) build
	@echo "✅ Package built successfully"

# Development workflow targets (legacy - use new combined commands above)
check-legacy: lint test-quick ## Quick development check (lint + unit tests)
validate-legacy: lint test-coverage ## Full validation (required before commit)

# CI/CD targets
ci-test: ## CI test pipeline
	$(MAKE) lint
	$(MAKE) test-coverage

# Help target showing Clean Code + TDD requirements
requirements: ## Show Clean Code + TDD requirements
	@echo "🎯 Clean Code + TDD Requirements:"
	@echo "  • Zero linting errors (make lint-check)"
	@echo "  • 100% line coverage (make test-coverage)"
	@echo "  • 100% branch coverage (--cov-branch)"
	@echo "  • Complexity < 5 (make metrics)"
	@echo "  • No unused code (make vulture)"
	@echo "  • Type safety (make type-check)"
	@echo "  • Security compliance (make security)"
	@echo "  • Methods < 20 lines"
	@echo "  • AAA test pattern"
	@echo "  • SOLID principles"
	@echo ""
	@echo "📋 Quality Tools:"
	@echo "  • make radon           # Code complexity analysis"
	@echo "  • make vulture         # Unused code detection"
	@echo "  • make type-check      # Static type verification"
	@echo "  • make static-analysis # Complete static analysis"
	@echo "  • make security        # Security analysis"
	@echo "  • make quality         # Complete quality check"
	@echo "  • make metrics         # Code metrics"
	@echo "  • make health          # Project health"
	@echo ""
	@echo "📋 Development Workflow:"
	@echo "  1. make dev-install    # Setup dev environment"
	@echo "  2. make check          # Quick validation during development" 
	@echo "  3. make validate       # Full validation before commit"
	@echo "  4. make build          # Build after validation"
	@echo ""
	@echo "💡 Usage: TARGET=path/to/file make [command] for specific files"
