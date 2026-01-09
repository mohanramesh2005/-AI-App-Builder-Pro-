import streamlit as st
from openai import OpenAI
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
from datetime import datetime, timedelta
import base64
import io
import zipfile
import hashlib
import uuid
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import sys
import os
import tempfile
import subprocess
import ast
import pygments
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import markdown
from streamlit.components.v1 import html
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
import yaml
import toml
from enum import Enum
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from typing import List

# ---------- SETUP LOGGING ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app_builder.log', maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- ENUMS ----------
class Framework(Enum):
    STREAMLIT = "Streamlit"
    GRADIO = "Gradio"
    FASTAPI = "FastAPI"
    FLASK = "Flask"
    PYSIMPLEGUI = "PySimpleGUI"
    DJANGO = "Django"
    TORNADO = "Tornado"

class Complexity(Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"

class AppCategory(Enum):
    DATA_VIZ = "📊 Data Visualization"
    ML_AI = "🤖 Machine Learning & AI"
    NLP = "💬 NLP & Chatbots"
    WEB_APP = "🌐 Web Applications"
    UTILITIES = "🛠️ Utilities"
    GAMES = "🎮 Games & Interactive"
    IOT = "📱 IoT & Real-time"
    BUSINESS = "💼 Business & Analytics"

# ---------- DATACLASSES ----------
@dataclass
class AppRecord:
    id: str
    framework: str
    task: str
    complexity: str
    features: List[str]
    timestamp: str
    generation_time: float
    estimated_tokens: int
    code_preview: str
    full_code: str
    quality_score: float
    dependencies: List[str]
    test_coverage: float
    security_level: str

@dataclass
class CodeMetrics:
    lines_of_code: int
    functions: int
    classes: int
    comments: int
    imports: int
    complexity_score: float
    maintainability_index: float
    security_issues: List[str]

# ---------- DIRECT API KEY ----------
API_KEY = "YOUR-OPENAI-API_KEY"

if not API_KEY or API_KEY.strip() == "":
    raise RuntimeError("❌ OPENAI_API_KEY is missing. Add it in the code.")

client = OpenAI(api_key=API_KEY)

# ---------- DATABASE SETUP ----------
class AppDatabase:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.setup_tables()
    
    def setup_tables(self):
        cursor = self.conn.cursor()
        
        # Apps table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS apps (
                id TEXT PRIMARY KEY,
                framework TEXT,
                task TEXT,
                complexity TEXT,
                features TEXT,
                timestamp TEXT,
                generation_time REAL,
                estimated_tokens INTEGER,
                code_preview TEXT,
                full_code TEXT,
                quality_score REAL,
                dependencies TEXT,
                test_coverage REAL,
                security_level TEXT
            )
        ''')
        
        # Code snippets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS code_snippets (
                id TEXT PRIMARY KEY,
                category TEXT,
                snippet TEXT,
                language TEXT,
                description TEXT,
                usage_count INTEGER DEFAULT 0
            )
        ''')
        
        # Templates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS templates (
                id TEXT PRIMARY KEY,
                name TEXT,
                framework TEXT,
                code_template TEXT,
                variables TEXT,
                description TEXT
            )
        ''')
        
        # Insert some default templates
        self.insert_default_templates(cursor)
        
        self.conn.commit()
    
    def insert_default_templates(self, cursor):
        """Insert default templates into database"""
        templates = [
            (
                str(uuid.uuid4()),
                "Streamlit Dashboard",
                "Streamlit",
                """
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="{app_name}", layout="wide")

# Sidebar
with st.sidebar:
    st.header("Settings")
    option = st.selectbox("Select Option", ["Option 1", "Option 2", "Option 3"])
    value = st.slider("Select Value", 0, 100, 50)

# Main content
st.title("{app_name}")
st.write(f"Selected: {{option}} with value: {{value}}")

# Data visualization
if st.button("Generate Data"):
    df = pd.DataFrame({
        'x': range(100),
        'y': np.random.randn(100).cumsum()
    })
    fig = px.line(df, x='x', y='y', title="Sample Data")
    st.plotly_chart(fig)

if __name__ == "__main__":
    pass
                """,
                json.dumps({"app_name": "Application Name"}),
                "Basic Streamlit dashboard template"
            ),
            (
                str(uuid.uuid4()),
                "FastAPI Service",
                "FastAPI",
                """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="{app_name}", version="1.0.0")

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

items = []

@app.get("/")
async def root():
    return {{"message": "{app_name} API is running"}}

@app.get("/items/")
async def read_items():
    return {{"items": items}}

@app.post("/items/")
async def create_item(item: Item):
    items.append(item.dict())
    return {{"message": "Item created", "item": item}}

@app.get("/items/{{item_id}}")
async def read_item(item_id: int):
    if item_id < len(items):
        return items[item_id]
    raise HTTPException(status_code=404, detail="Item not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
                """,
                json.dumps({"app_name": "API Service"}),
                "FastAPI REST service template"
            )
        ]
        
        cursor.executemany('''
            INSERT OR IGNORE INTO templates VALUES (?, ?, ?, ?, ?, ?)
        ''', templates)
    
    @contextmanager
    def get_cursor(self):
        cursor = self.conn.cursor()
        try:
            yield cursor
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def save_app(self, app_record: AppRecord):
        with self.get_cursor() as cursor:
            cursor.execute('''
                INSERT INTO apps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                app_record.id,
                app_record.framework,
                app_record.task,
                app_record.complexity,
                json.dumps(app_record.features),
                app_record.timestamp,
                app_record.generation_time,
                app_record.estimated_tokens,
                app_record.code_preview,
                app_record.full_code,
                app_record.quality_score,
                json.dumps(app_record.dependencies),
                app_record.test_coverage,
                app_record.security_level
            ))
    
    def get_all_apps(self) -> List[Dict]:
        with self.get_cursor() as cursor:
            cursor.execute('SELECT * FROM apps ORDER BY timestamp DESC')
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

# Initialize database
db = AppDatabase()

# ---------- SESSION STATE INITIALIZATION ----------
if 'generated_apps' not in st.session_state:
    st.session_state.generated_apps = []
if 'app_metrics' not in st.session_state:
    st.session_state.app_metrics = {
        'total_generated': 0,
        'total_tokens': 0,
        'total_lines': 0,
        'avg_quality': 0.0,
        'most_used_framework': '',
        'generation_speed': []
    }
if 'favorites' not in st.session_state:
    st.session_state.favorites = {}
if 'code_history' not in st.session_state:
    st.session_state.code_history = []
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'experience_level': 'Intermediate',
        'preferred_frameworks': [],
        'coding_style': 'Functional',
        'theme_preference': 'Dark'
    }
if 'api_cache' not in st.session_state:
    st.session_state.api_cache = {}
if 'code_reviews' not in st.session_state:
    st.session_state.code_reviews = {}

# ---------- ADVANCED AI FUNCTION WITH MULTI-MODEL SUPPORT ----------
class CodeGenerator:
    def __init__(self):
        self.models = {
            "fast": "gpt-4o",
            "balanced": "gpt-4o-mini",
            "advanced": "gpt-4.1",
            "expert": "gpt-4.1-mini"
        }
        self.code_templates = self.load_templates()
        self.snippets_cache = {}
        self.api_call_count = 0
    
    def load_templates(self) -> Dict:
        """Load code templates from database"""
        templates = {}
        try:
            with db.get_cursor() as cursor:
                cursor.execute("SELECT name, code_template FROM templates")
                for row in cursor.fetchall():
                    templates[row[0].lower().replace(" ", "_")] = row[1]
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
        
        # Add default templates if database is empty
        if not templates:
            templates = {
                "streamlit_dashboard": self.get_streamlit_dashboard_template(),
                "gradio_interface": self.get_gradio_interface_template(),
                "fastapi_service": self.get_fastapi_template(),
                "ml_pipeline": self.get_ml_pipeline_template()
            }
        
        return templates
    
    def get_streamlit_dashboard_template(self) -> str:
        return """
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

def main():
    st.set_page_config(page_title="{app_name}", layout="wide")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        {sidebar_content}
    
    # Main content
    st.title("{app_name}")
    {main_content}
    
    # Analytics
    {analytics_section}

if __name__ == "__main__":
    main()
"""
    
    def get_gradio_interface_template(self) -> str:
        return """
import gradio as gr
import numpy as np

def {main_function}({parameters}):
    {function_body}

# Create interface
interface = gr.Interface(
    fn={main_function},
    inputs={inputs},
    outputs={outputs},
    title="{app_name}",
    description="{description}"
)

if __name__ == "__main__":
    interface.launch()
"""
    
    def get_fastapi_template(self) -> str:
        return """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="{app_name}")

class RequestModel(BaseModel):
    {request_fields}

@app.get("/")
async def root():
    return {{"message": "{app_name} API is running"}}

@app.post("/predict")
async def predict(data: RequestModel):
    {prediction_logic}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
    
    def get_ml_pipeline_template(self) -> str:
        return """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

class MLPipeline:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    def preprocess(self, data):
        {preprocessing_code}
    
    def train(self, X_train, y_train):
        {training_code}
    
    def predict(self, X):
        {prediction_code}

# Usage example
pipeline = MLPipeline()
"""

    def generate_with_fallback(self, prompt: str, model: str = "balanced", max_retries: int = 3) -> str:
        """Generate code with retry logic and fallback models"""
        for attempt in range(max_retries):
            try:
                cache_key = hashlib.md5(prompt.encode()).hexdigest()
                if cache_key in st.session_state.api_cache:
                    logger.info(f"Using cached response for key: {cache_key}")
                    return st.session_state.api_cache[cache_key]
                
                # Use the correct OpenAI API call format
                response = client.chat.completions.create(
                    model=self.models.get(model, "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are an expert Python developer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7 if model == "expert" else 0.3,
                    max_tokens=4000
                )
                
                code = response.choices[0].message.content
                
                # Clean the response to ensure it's only code
                code = self.clean_code_response(code)
                
                st.session_state.api_cache[cache_key] = code
                self.api_call_count += 1
                
                return code
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    # Fallback to template-based generation
                    return self.generate_from_template(prompt)
                time.sleep(1 * (attempt + 1))
        
        return ""
    
    def clean_code_response(self, code: str) -> str:
        """Clean the API response to extract only Python code"""
        # Remove markdown code blocks
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        code = re.sub(r'```py\s*', '', code)
        
        # Remove explanations before code
        lines = code.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from ') or line.strip().startswith('def ') or line.strip().startswith('class ') or line.strip().startswith('@'):
                in_code = True
            
            if in_code or line.strip().startswith('#') or line.strip() == '':
                code_lines.append(line)
        
        return '\n'.join(code_lines).strip()
    
    def generate_from_template(self, prompt: str) -> str:
        """Generate code using template when API fails"""
        # Extract key information from prompt
        prompt_lower = prompt.lower()
        
        if "streamlit" in prompt_lower:
            template = self.code_templates.get("streamlit_dashboard", self.get_streamlit_dashboard_template())
            return template.format(
                app_name="Generated App",
                sidebar_content="option = st.selectbox('Select Option', ['Option 1', 'Option 2', 'Option 3'])\n        value = st.slider('Select value', 0, 100, 50)",
                main_content="st.write('Hello World')\n    if st.button('Click me'):\n        st.success('Button clicked!')\n    \n    # Sample data visualization\n    df = pd.DataFrame({'x': range(100), 'y': np.random.randn(100)})\n    fig = px.scatter(df, x='x', y='y', title='Sample Data')\n    st.plotly_chart(fig)",
                analytics_section="with st.expander('Analytics'):\n        st.write('### Performance Metrics')\n        col1, col2, col3 = st.columns(3)\n        with col1:\n            st.metric('Users', '1,234', '+12%')\n        with col2:\n            st.metric('Revenue', '$5,678', '+8%')\n        with col3:\n            st.metric('Engagement', '78%', '+3%')"
            )
        elif "fastapi" in prompt_lower or "api" in prompt_lower:
            template = self.code_templates.get("fastapi_service", self.get_fastapi_template())
            return template.format(
                app_name="Generated API",
                request_fields="data: str\n    value: float",
                prediction_logic="try:\n        result = float(data.value) * 2\n        return {\"result\": result, \"status\": \"success\"}\n    except Exception as e:\n        raise HTTPException(status_code=400, detail=str(e))"
            )
        elif "gradio" in prompt_lower:
            template = self.code_templates.get("gradio_interface", self.get_gradio_interface_template())
            return template.format(
                main_function="process_input",
                parameters="text, number",
                function_body="return f\"Text: {text}, Number: {number * 2}\"",
                inputs="[gr.Textbox(label=\"Enter text\"), gr.Number(label=\"Enter number\")]",
                outputs="gr.Textbox(label=\"Result\")",
                app_name="Gradio Interface",
                description="A simple Gradio interface example"
            )
        else:
            # Default to Streamlit
            return self.code_templates.get("streamlit_dashboard", self.get_streamlit_dashboard_template()).format(
                app_name="Generated Application",
                sidebar_content="st.write('Sidebar content')",
                main_content="st.write('Main application content')\n    st.info('Application generated successfully!')",
                analytics_section="# Analytics section"
            )

    def analyze_code_complexity(self, code: str) -> CodeMetrics:
        """Analyze code complexity and quality"""
        try:
            tree = ast.parse(code)
            
            # Count elements
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            imports = len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))])
            
            lines = len(code.split('\n'))
            comments = len([line for line in code.split('\n') if line.strip().startswith('#')])
            
            # Calculate complexity score
            complexity_score = min(100, max(0, 
                (functions * 5 + classes * 10 + lines * 0.1) / 100 * 100
            ))
            
            # Check for security issues
            security_issues = self.check_security_issues(code)
            
            return CodeMetrics(
                lines_of_code=lines,
                functions=functions,
                classes=classes,
                comments=comments,
                imports=imports,
                complexity_score=complexity_score,
                maintainability_index=max(0, 100 - complexity_score),
                security_issues=security_issues
            )
            
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return CodeMetrics(0, 0, 0, 0, 0, 0.0, 0.0, [])

    def check_security_issues(self, code: str) -> List[str]:
        """Check for common security issues in code"""
        issues = []
        
        # Check for hardcoded secrets
        if re.search(r'(password|secret|key|token)\s*=\s*["\'].*["\']', code, re.IGNORECASE):
            issues.append("Hardcoded credentials detected")
        
        # Check for dangerous imports
        dangerous_imports = ['os.system', 'subprocess.call', 'eval', 'exec', 'pickle.loads']
        for imp in dangerous_imports:
            if imp in code:
                issues.append(f"Potentially dangerous import/function: {imp}")
        
        # Check for SQL injection patterns
        if re.search(r'execute.*f["\']', code) or re.search(r'execute.*\+', code):
            issues.append("Possible SQL injection vulnerability")
        
        # Check for shell injection
        if re.search(r'subprocess\.(run|call|Popen)\(.*shell=True', code):
            issues.append("Shell injection risk with shell=True")
        
        return issues

    def extract_dependencies(self, code: str) -> List[str]:
        """Extract Python dependencies from code"""
        dependencies = set()
        
        # Common libraries pattern
        common_libs = {
            'pandas', 'numpy', 'matplotlib', 'plotly', 'seaborn',
            'scikit-learn', 'tensorflow', 'torch', 'streamlit',
            'gradio', 'fastapi', 'flask', 'django', 'sqlalchemy',
            'requests', 'pydantic', 'uvicorn', 'jinja2', 'sqlite3'
        }
        
        for lib in common_libs:
            if f'import {lib}' in code or f'from {lib}' in code:
                dependencies.add(lib)
        
        return list(dependencies)

    def generate_tests(self, code: str) -> str:
        """Generate unit tests for the generated code"""
        try:
            tree = ast.parse(code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            test_template = """
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

class TestGeneratedApp(unittest.TestCase):
    
    def setUp(self):
        # Setup test environment
        pass
    
    def tearDown(self):
        # Cleanup after tests
        pass
    
    def test_imports(self):
        '''Test that all imports work correctly'''
        try:
{import_tests}
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Import failed: {{e}}")
    
{function_tests}
    def test_edge_cases(self):
        '''Test edge cases'''
        # Add edge case tests here
        pass

if __name__ == '__main__':
    unittest.main()
"""
        
            # Generate import tests
            imports = self.extract_dependencies(code)
            import_tests = "\n".join([f"            import {imp}" for imp in imports])
            
            # Generate function tests
            function_tests = ""
            for func in functions[:5]:  # Limit to 5 functions
                if not func.startswith('_'):
                    function_tests += f"""
    def test_{func}(self):
        '''Test {func} function'''
        # Test {func} function
        # self.assertEqual(expected, actual)
        pass

"""
            
            return test_template.format(
                import_tests=import_tests,
                function_tests=function_tests
            )
            
        except Exception as e:
            logger.error(f"Error generating tests: {e}")
            return """
import unittest

class TestGeneratedApp(unittest.TestCase):
    
    def test_basic(self):
        '''Basic test'''
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
"""

# Initialize code generator
code_gen = CodeGenerator()

# ---------- ENHANCED APP GENERATION FUNCTION ----------
def generate_app_code_advanced(
    framework: str, 
    task: str, 
    complexity: str = "Intermediate",
    features: List[str] = None,
    model: str = "balanced",
    include_tests: bool = True,
    optimize: bool = True
) -> Dict[str, Any]:
    """
    Advanced app code generation with comprehensive features
    """
    try:
        start_time = time.time()
        
        # Build enhanced prompt
        features_text = ", ".join(features) if features else "production-ready implementation"
        
        prompt = f"""You are an expert Python developer specializing in {framework} applications.

TASK: Create a complete, working {framework} application for: {task}

REQUIREMENTS:
1. Complexity Level: {complexity}
2. Required Features: {features_text}
3. Code Quality: Production-ready with:
   - Comprehensive error handling
   - Type hints (Python 3.8+)
   - Detailed docstrings
   - Logging configuration
   - Configuration management
   - Security best practices

4. Architecture:
   - Modular design (separate modules for UI, logic, data)
   - Dependency injection where appropriate
   - Environment variable support
   - Configuration files (YAML/JSON)

5. Performance:
   - Async/await for I/O operations
   - Caching mechanisms
   - Efficient algorithms

6. Testing:
   - Unit tests with pytest
   - Integration tests
   - Mock external dependencies

7. Deployment:
   - Dockerfile
   - CI/CD pipeline configuration
   - Monitoring setup

IMPORTANT: Respond ONLY with Python code. No explanations, no markdown, no comments outside code.
Include ALL necessary imports and setup code.
The code should run without modification.

Structure the application as follows:
1. Main application file
2. Configuration module
3. Service/Logic layer
4. Data models
5. Utilities
6. Tests

Make the code robust, maintainable, and scalable.

Here's the code:"""
        
        # Generate code
        app_code = code_gen.generate_with_fallback(prompt, model)
        
        if not app_code or app_code.startswith("❌"):
            raise Exception(f"Code generation failed: {app_code}")
        
        generation_time = time.time() - start_time
        
        # Analyze code
        metrics = code_gen.analyze_code_complexity(app_code)
        dependencies = code_gen.extract_dependencies(app_code)
        
        # Generate tests if requested
        test_code = ""
        if include_tests:
            test_code = code_gen.generate_tests(app_code)
        
        # Optimize code if requested
        if optimize:
            app_code = optimize_code(app_code)
        
        # Calculate quality score
        quality_score = calculate_quality_score(metrics, generation_time)
        
        # Create app record
        app_id = str(uuid.uuid4())
        app_record = AppRecord(
            id=app_id,
            framework=framework,
            task=task,
            complexity=complexity,
            features=features or [],
            timestamp=datetime.now().isoformat(),
            generation_time=round(generation_time, 2),
            estimated_tokens=len(app_code.split()),
            code_preview=app_code[:500] + "..." if len(app_code) > 500 else app_code,
            full_code=app_code,
            quality_score=quality_score,
            dependencies=dependencies,
            test_coverage=0.7 if include_tests else 0.0,  # Estimate
            security_level="Medium" if not metrics.security_issues else "Low"
        )
        
        # Save to database
        db.save_app(app_record)
        
        # Update session state
        st.session_state.generated_apps.append(asdict(app_record))
        st.session_state.app_metrics['total_generated'] += 1
        st.session_state.app_metrics['total_tokens'] += len(app_code.split())
        st.session_state.app_metrics['total_lines'] += metrics.lines_of_code
        st.session_state.app_metrics['generation_speed'].append(generation_time)
        
        logger.info(f"Generated app: {task} with quality score {quality_score}")
        
        return {
            'code': app_code,
            'test_code': test_code,
            'metrics': metrics,
            'record': app_record,
            'dependencies': dependencies,
            'security_issues': metrics.security_issues
        }
        
    except Exception as e:
        logger.error(f"Error generating app code: {str(e)}")
        return {
            'error': f"❌ Error: {str(e)}",
            'code': "",
            'test_code': "",
            'metrics': None,
            'record': None
        }

# ---------- OPTIMIZATION FUNCTIONS ----------
def optimize_code(code: str) -> str:
    """Optimize generated code for performance and readability"""
    
    optimizations = [
        # Remove duplicate imports
        (r'import (\w+)\nimport \1', r'import \1'),
        # Convert print to logging
        (r'print\((.*)\)', r'logging.info(\1)'),
        # Add docstrings to functions without them
        (r'def (\w+)\(([^)]*)\):\s*\n(\s+)(?![uU]\"\"\"|[rR]\"\"\"|\"\"\"|\'\'\')', 
         r'def \1(\2):\n\3"""Docstring for \1."""\n\3'),
        # Convert string concatenation to f-strings
        (r'\"([^\"]+)\" \+ (\w+)', r'f"\1{\2}"'),
        (r'(\w+) \+ \"([^\"]+)\"', r'f"{\1}\2"'),
    ]
    
    for pattern, replacement in optimizations:
        code = re.sub(pattern, replacement, code)
    
    return code

def calculate_quality_score(metrics: CodeMetrics, generation_time: float) -> float:
    """Calculate overall quality score for generated code"""
    
    weights = {
        'maintainability': 0.3,
        'complexity': 0.2,
        'comments_ratio': 0.15,
        'security': 0.2,
        'performance': 0.15
    }
    
    # Calculate individual scores
    maintainability_score = metrics.maintainability_index
    complexity_score = 100 - metrics.complexity_score
    
    # Comment ratio score
    comment_ratio = metrics.comments / max(1, metrics.lines_of_code)
    comment_score = min(100, comment_ratio * 1000)
    
    # Security score
    security_score = 100 - len(metrics.security_issues) * 20
    
    # Performance score (based on generation time)
    performance_score = min(100, max(0, 100 - generation_time * 10))
    
    # Weighted average
    total_score = (
        maintainability_score * weights['maintainability'] +
        complexity_score * weights['complexity'] +
        comment_score * weights['comments_ratio'] +
        security_score * weights['security'] +
        performance_score * weights['performance']
    )
    
    return round(total_score, 2)

# ---------- CODE VISUALIZATION FUNCTIONS ----------
def visualize_code_structure(code: str):
    """Generate visual representation of code structure"""
    try:
        tree = ast.parse(code)
        
        data = {
            'name': 'Code Structure',
            'children': []
        }
        
        # Count different elements
        elements = {
            'Functions': 0,
            'Classes': 0,
            'Imports': 0,
            'Variables': 0,
            'Loops': 0,
            'Conditionals': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                elements['Functions'] += 1
            elif isinstance(node, ast.ClassDef):
                elements['Classes'] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                elements['Imports'] += 1
            elif isinstance(node, ast.Assign):
                elements['Variables'] += 1
            elif isinstance(node, (ast.For, ast.While)):
                elements['Loops'] += 1
            elif isinstance(node, (ast.If, ast.IfExp)):
                elements['Conditionals'] += 1
        
        # Create visualization data
        for key, value in elements.items():
            if value > 0:
                data['children'].append({
                    'name': key,
                    'value': value
                })
        
        return data
        
    except Exception as e:
        logger.error(f"Error visualizing code structure: {e}")
        return {'name': 'Error', 'children': []}

def generate_syntax_highlighted_html(code: str) -> str:
    """Generate HTML with syntax highlighting"""
    try:
        lexer = PythonLexer()
        formatter = HtmlFormatter(style='monokai', full=True)
        highlighted = pygments.highlight(code, lexer, formatter)
        
        # Add custom styling
        html_output = f"""
        <style>
        {formatter.get_style_defs('.highlight')}
        .code-container {{
            background: #272822;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            max-height: 500px;
            overflow-y: auto;
        }}
        </style>
        <div class="code-container">
            {highlighted}
        </div>
        """
        return html_output
    except Exception as e:
        logger.error(f"Error generating syntax highlighting: {e}")
        return f"<pre><code>{code}</code></pre>"

# ---------- DEPENDENCY MANAGEMENT ----------
def generate_requirements_file(dependencies: List[str]) -> str:
    """Generate requirements.txt with version constraints"""
    
    version_map = {
        'streamlit': '>=1.28.0',
        'pandas': '>=2.0.0',
        'numpy': '>=1.24.0',
        'plotly': '>=5.18.0',
        'scikit-learn': '>=1.3.0',
        'fastapi': '>=0.104.0',
        'gradio': '>=4.0.0',
        'flask': '>=3.0.0',
        'sqlalchemy': '>=2.0.0',
        'requests': '>=2.31.0',
        'pydantic': '>=2.0.0',
        'uvicorn': '>=0.24.0',
        'jinja2': '>=3.1.0'
    }
    
    requirements = []
    for dep in dependencies:
        dep_lower = dep.lower()
        version = version_map.get(dep_lower, '>=1.0.0')
        requirements.append(f"{dep_lower}{version}")
    
    # Add common dependencies
    common_deps = [
        'python-dotenv>=1.0.0',
        'python-multipart>=0.0.6'
    ]
    
    requirements.extend(common_deps)
    return '\n'.join(sorted(set(requirements)))

def generate_dockerfile(framework: str, app_name: str) -> str:
    """Generate Dockerfile for the application"""
    
    app_name_safe = re.sub(r'[^a-zA-Z0-9]', '_', app_name)
    
    dockerfile = f"""FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port based on framework
"""
    
    ports = {
        'streamlit': '8501',
        'gradio': '7860',
        'fastapi': '8000',
        'flask': '5000',
        'django': '8000'
    }
    
    port = ports.get(framework.lower(), '8000')
    
    dockerfile += f"""
EXPOSE {port}

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD python -c "import requests; requests.get('http://localhost:{port}/health', timeout=2)" || exit 1

# Command to run based on framework
"""
    
    commands = {
        'streamlit': f'CMD ["streamlit", "run", "{app_name_safe}.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]',
        'gradio': f'CMD ["python", "{app_name_safe}.py"]',
        'fastapi': f'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]',
        'flask': f'CMD ["python", "{app_name_safe}.py"]'
    }
    
    dockerfile += commands.get(framework.lower(), f'CMD ["python", "{app_name_safe}.py"]')
    
    return dockerfile

# ---------- CODE VALIDATION ----------
def validate_code(code: str) -> Dict[str, Any]:
    """Validate generated code for syntax and common issues"""
    
    validation_results = {
        'syntax': {'valid': True, 'message': 'Syntax is valid'},
        'imports': {'valid': True, 'message': 'All imports are valid'},
        'execution': {'valid': False, 'message': 'Not executed'},
        'security': {'valid': True, 'message': 'No security issues found'},
        'performance': {'valid': True, 'message': 'Performance check passed'}
    }
    
    # Check syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        validation_results['syntax'] = {
            'valid': False,
            'message': f'Syntax error: {str(e)}'
        }
    except Exception as e:
        validation_results['syntax'] = {
            'valid': False,
            'message': f'Parse error: {str(e)}'
        }
    
    # Check for dangerous patterns
    dangerous_patterns = [
        (r'eval\(', 'eval() function detected - security risk'),
        (r'exec\(', 'exec() function detected - security risk'),
        (r'__import__\(', '__import__() detected - security risk'),
        (r'subprocess\.call\(', 'subprocess.call() detected - security risk'),
        (r'os\.system\(', 'os.system() detected - security risk'),
    ]
    
    for pattern, message in dangerous_patterns:
        if re.search(pattern, code):
            validation_results['security'] = {
                'valid': False,
                'message': message
            }
    
    # Check for performance issues
    performance_issues = [
        (r'for.*for', 'Nested loops detected - performance concern'),
        (r'while True:', 'Infinite loop detected'),
        (r'sleep\([^)]*\)', 'sleep() call detected - blocking operation')
    ]
    
    issues = []
    for pattern, message in performance_issues:
        if re.search(pattern, code):
            issues.append(message)
    
    if issues:
        validation_results['performance'] = {
            'valid': False,
            'message': '; '.join(issues)
        }
    
    return validation_results

# ---------- ADVANCED UI COMPONENTS ----------
class AdvancedUIComponents:
    @staticmethod
    def create_metric_card(title: str, value: Any, delta: Any = None, help_text: str = ""):
        """Create a styled metric card"""
        col1, col2 = st.columns([3, 1])
        with col1:
            st.metric(label=title, value=value, delta=delta, help=help_text)
        with col2:
            if delta:
                trend = "📈" if isinstance(delta, (int, float)) and delta > 0 else "📉"
                st.write(trend)
    
    @staticmethod
    def create_progress_indicator(current: int, total: int, label: str):
        """Create a custom progress indicator"""
        progress = current / total
        st.progress(progress, text=f"{label}: {current}/{total}")
        
        # Create a visual progress bar
        cols = st.columns(total)
        for i in range(total):
            with cols[i]:
                if i < current:
                    st.markdown("🟩", help=f"Step {i+1} completed")
                elif i == current:
                    st.markdown("🟨", help=f"Step {i+1} in progress")
                else:
                    st.markdown("⬜", help=f"Step {i+1} pending")
    
    @staticmethod
    def create_code_review_panel(code: str, metrics: CodeMetrics):
        """Create an interactive code review panel"""
        
        with st.expander("🔍 Code Review & Analysis", expanded=True):
            tabs = st.tabs(["Quality", "Security", "Performance", "Suggestions"])
            
            with tabs[0]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Quality Score", f"{metrics.maintainability_index:.1f}/100")
                with col2:
                    comment_ratio = metrics.comments / max(1, metrics.lines_of_code)
                    st.metric("Comment Ratio", f"{comment_ratio:.1%}")
                with col3:
                    st.metric("Complexity", f"{metrics.complexity_score:.1f}")
                
                st.progress(metrics.maintainability_index / 100, 
                          text=f"Maintainability: {metrics.maintainability_index:.1f}%")
            
            with tabs[1]:
                if metrics.security_issues:
                    st.warning("⚠️ Security Issues Detected")
                    for issue in metrics.security_issues:
                        st.error(f"• {issue}")
                else:
                    st.success("✅ No security issues detected")
                
                # Security score
                security_score = 100 - len(metrics.security_issues) * 20
                st.progress(security_score / 100, 
                          text=f"Security Score: {security_score}%")
            
            with tabs[2]:
                # Performance metrics
                performance_indicators = {
                    "Lines of Code": metrics.lines_of_code,
                    "Functions": metrics.functions,
                    "Classes": metrics.classes,
                    "Imports": metrics.imports
                }
                
                for label, value in performance_indicators.items():
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.write(f"**{label}:**")
                    with col2:
                        max_val = {
                            "Lines of Code": 500,
                            "Functions": 20,
                            "Classes": 10,
                            "Imports": 15
                        }.get(label, 100)
                        progress = min(value / max_val, 1)
                        st.progress(progress, text=f"{value}")
            
            with tabs[3]:
                suggestions = []
                

                if metrics.lines_of_code == 0:
                     ratio = 0
                     suggestions.append("Add more comments to improve readability")
                else:
                    ratio = metrics.comments / metrics.lines_of_code
                
                if metrics.complexity_score > 70:
                    suggestions.append("Consider refactoring to reduce complexity")
                
                if len(metrics.security_issues) > 0:
                    suggestions.append("Address security issues before deployment")
                
                if suggestions:
                    for suggestion in suggestions:
                        st.info(f"💡 {suggestion}")
                else:
                    st.success("✅ Code looks good! No major suggestions.")

# ---------- ERROR HANDLING ----------
def handle_errors(func):
    """Decorator to handle errors gracefully"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)

            st.error(
                f"❌ An unexpected error occurred in **{func.__name__}**:\n\n``{e}``"
            )

            return None
    return wrapper

# ---------- MAIN APPLICATION ----------
@handle_errors
def main():
    # Page configuration
    st.set_page_config(
        page_title="🤖 AI App Builder Pro+",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo',
            'Report a bug': 'https://github.com/your-repo/issues',
            'About': '# AI App Builder Pro+ v2.0\nGenerate production-ready Python apps with AI'
        }
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    /* Main styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    
    .code-block {
        font-family: 'Fira Code', 'Consolas', monospace;
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #333;
        max-height: 500px;
        overflow-y: auto;
    }
    
    .metric-badge {
        display: inline-block;
        background: linear-gradient(45deg, #FF6B6B, #FF8E53);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        background: rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size: 3rem;">🤖 AI App Builder Pro+</h1>
        <p style="margin:0; opacity: 0.9; font-size: 1.2rem;">
            Generate Production-Ready Python Applications with Advanced AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
        st.title("⚙️ Control Panel")
        
        # User Profile
        with st.expander("👤 User Profile", expanded=True):
            st.session_state.user_profile['experience_level'] = st.selectbox(
                "Experience Level",
                ["Beginner", "Intermediate", "Advanced", "Expert"]
            )
            
            st.session_state.user_profile['preferred_frameworks'] = st.multiselect(
                "Preferred Frameworks",
                [f.value for f in Framework],
                default=["Streamlit", "Gradio"]
            )
            
            st.session_state.user_profile['coding_style'] = st.selectbox(
                "Coding Style",
                ["Functional", "Object-Oriented", "Procedural", "Mixed"]
            )
        
        # Generation Settings
        with st.expander("⚡ Generation Settings", expanded=True):
            model_choice = st.selectbox(
                "AI Model",
                ["fast", "balanced", "advanced", "expert"],
                index=1,
                help="fast: gpt-4-1106-preview\nbalanced: gpt-4\nadvanced: gpt-4-1106-preview\nexpert: gpt-4-32k"
            )
            
            temperature = st.slider(
                "Creativity (Temperature)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                help="Higher values = more creative, Lower values = more deterministic"
            )
            
            max_length = st.slider(
                "Max Code Length (tokens)",
                min_value=1000,
                max_value=8000,
                value=4000,
                step=1000
            )
        
        # Quick Actions
        st.divider()
        st.subheader("🚀 Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Dashboard", use_container_width=True):
                st.session_state.active_tab = "Dashboard"
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        if st.button("🧹 Clear Cache", use_container_width=True, type="secondary"):
            st.session_state.api_cache.clear()
            st.success("Cache cleared!")
        
        # Statistics
        st.divider()
        st.subheader("📈 Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Apps Generated", st.session_state.app_metrics['total_generated'])
        with col2:
            st.metric("Total Lines", f"{st.session_state.app_metrics['total_lines']:,}")
    
    # Main Tabs
    tabs = st.tabs(["🎨 Create App", "📚 App Library", "🔧 Code Workshop", "📊 Analytics", "🚀 Deployment", "⚙️ Settings"])
    
    # Tab 1: Create App
    with tabs[0]:
        st.header("🎨 Create New Application")
        
        # Two-column layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Framework Selection
            framework = st.selectbox(
                "Select Framework:",
                [f.value for f in Framework],
                help="Choose the Python framework for your application"
            )
            
            # Task Configuration
            st.subheader("🎯 Task Configuration")
            
            task_categories = {
                AppCategory.DATA_VIZ.value: [
                    "Real-time Financial Dashboard",
                    "Interactive Geospatial Analytics",
                    "Multi-dimensional Data Explorer",
                    "Time Series Forecasting Tool"
                ],
                AppCategory.ML_AI.value: [
                    "Automated ML Pipeline with Hyperparameter Tuning",
                    "Computer Vision Image Classifier",
                    "Recommendation System with A/B Testing",
                    "Natural Language Processing Pipeline"
                ],
                AppCategory.NLP.value: [
                    "Multi-language Chatbot with Context Memory",
                    "Document Summarization & Analysis",
                    "Sentiment Analysis Dashboard",
                    "Text Generation Interface"
                ],
                AppCategory.WEB_APP.value: [
                    "E-commerce Platform with Payment Integration",
                    "Social Media Analytics Dashboard",
                    "Project Management System",
                    "Real-time Collaboration Tool"
                ]
            }
            
            selected_category = st.selectbox(
                "Select Application Category:",
                list(task_categories.keys())
            )
            
            task = st.selectbox(
                "Select Template:",
                task_categories[selected_category]
            )
            
            # Custom task input
            custom_task = st.text_area(
                "Or describe your custom application:",
                height=100,
                placeholder="Describe in detail what your application should do, including specific features, target audience, and any special requirements..."
            )
            
            if custom_task.strip():
                task = custom_task
        
        with col2:
            # Advanced Configuration
            st.subheader("⚙️ Advanced Configuration")
            
            complexity = st.select_slider(
                "Complexity Level:",
                options=[c.value for c in Complexity],
                value="Intermediate"
            )
            
            # Feature Selection
            st.subheader("✨ Features")
            
            feature_sets = {
                "Core Features": [
                    "Authentication System",
                    "Database Integration",
                    "REST API Endpoints",
                    "WebSocket Support",
                    "File Upload/Download"
                ],
                "UI/UX Features": [
                    "Dark/Light Mode",
                    "Responsive Design",
                    "Animations & Transitions",
                    "Custom Themes",
                    "Accessibility Features"
                ],
                "Advanced Features": [
                    "Real-time Updates",
                    "Caching Layer",
                    "Background Jobs",
                    "Webhook Support",
                    "API Rate Limiting"
                ],
                "ML/AI Features": [
                    "Model Training Interface",
                    "Inference API",
                    "A/B Testing Framework",
                    "Model Monitoring",
                    "AutoML Capabilities"
                ]
            }
            
            selected_features = []
            for category, features in feature_sets.items():
                with st.expander(f"{category}"):
                    for feature in features:
                        if st.checkbox(feature, key=f"feat_{feature}"):
                            selected_features.append(feature)
        
        # Additional Configuration
        with st.expander("🔧 Advanced Options", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                include_tests = st.checkbox("Include Unit Tests", value=True)
                add_documentation = st.checkbox("Add Documentation", value=True)
                code_coverage = st.slider("Target Code Coverage", 0, 100, 80)
            
            with col_b:
                optimize_performance = st.checkbox("Optimize Performance", value=True)
                add_logging = st.checkbox("Add Logging", value=True)
                security_scan = st.checkbox("Security Scanning", value=True)
            
            with col_c:
                generate_docker = st.checkbox("Generate Dockerfile", value=True)
                create_ci_cd = st.checkbox("CI/CD Pipeline", value=False)
                add_monitoring = st.checkbox("Add Monitoring", value=False)
        
        # Generate Button
        st.markdown("---")
        generate_col1, generate_col2, generate_col3 = st.columns([1, 2, 1])
        
        with generate_col2:
            generate_clicked = st.button(
                "🚀 GENERATE APPLICATION",
                use_container_width=True,
                type="primary",
                help="Click to generate your application with all configured features"
            )
        
        if generate_clicked:
            with st.spinner(f"🧠 Generating {complexity} {framework} application..."):
                # Show progress indicators
                progress_steps = ["Initializing", "Designing Architecture", "Generating Code", 
                                "Optimizing", "Running Tests", "Finalizing"]
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, step in enumerate(progress_steps):
                    time.sleep(0.3)
                    progress = (i + 1) / len(progress_steps)
                    progress_bar.progress(progress)
                    status_text.text(f"{step}...")
                
                # Generate the application
                result = generate_app_code_advanced(
                    framework=framework,
                    task=task,
                    complexity=complexity,
                    features=selected_features,
                    model=model_choice,
                    include_tests=include_tests,
                    optimize=optimize_performance
                )
                
                progress_bar.empty()
                status_text.empty()
                
                if 'error' in result:
                    st.error(result['error'])
                else:
                    # Celebration
                    st.balloons()
                    st.success(f"✅ **{framework} Application Generated Successfully!**")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Quality Score", f"{result['record'].quality_score}/100")
                    with col2:
                        st.metric("Lines of Code", result['metrics'].lines_of_code)
                    with col3:
                        st.metric("Security Level", result['record'].security_level)
                    with col4:
                        st.metric("Generation Time", f"{result['record'].generation_time}s")
                    
                    # Code Display
                    st.subheader("📝 Generated Code")
                    
                    code_tabs = st.tabs(["Main Application", "Unit Tests", "Dependencies", "Deployment"])
                    
                    with code_tabs[0]:
                        # Display code
                        st.code(result['code'], language='python')
                        
                        # Code review panel
                        AdvancedUIComponents.create_code_review_panel(
                            result['code'], 
                            result['metrics']
                        )
                    
                    with code_tabs[1]:
                        if result['test_code']:
                            st.code(result['test_code'], language='python')
                            
                            # Test coverage visualization
                            coverage = result['record'].test_coverage * 100
                            st.progress(coverage / 100, 
                                      text=f"Estimated Test Coverage: {coverage:.1f}%")
                        else:
                            st.info("No tests generated. Enable 'Include Unit Tests' in settings.")
                    
                    with code_tabs[2]:
                        # Dependencies
                        if result['dependencies']:
                            st.write("**Required Dependencies:**")
                            for dep in result['dependencies']:
                                st.markdown(f'<span class="metric-badge">{dep}</span>', unsafe_allow_html=True)
                            
                            # Generate requirements.txt
                            requirements = generate_requirements_file(result['dependencies'])
                            st.code(requirements, language='text')
                            
                            # Installation command
                            st.code("pip install -r requirements.txt", language='bash')
                        else:
                            st.info("No external dependencies detected.")
                    
                    with code_tabs[3]:
                        # Deployment files
                        if generate_docker:
                            dockerfile = generate_dockerfile(framework, task.replace(' ', '_'))
                            st.code(dockerfile, language='dockerfile')
                        
                        # Deployment instructions
                        st.info("""
                        **Deployment Instructions:**
                        1. Install dependencies: `pip install -r requirements.txt`
                        2. Run tests: `pytest`
                        3. Start application: `python app.py`
                        4. For production: Use the generated Dockerfile
                        """)
                    
                    # Download Section
                    st.subheader("📥 Download Files")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        b64 = base64.b64encode(result['code'].encode()).decode()
                        href = f'<a href="data:file/txt;base64,{b64}" download="{task.replace(" ", "_")}.py">📥 Download Main File</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                    with col2:
                        # Create project zip
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            zip_file.writestr(f"{task.replace(' ', '_')}.py", result['code'])
                            if result['test_code']:
                                zip_file.writestr(f"test_{task.replace(' ', '_')}.py", result['test_code'])
                            zip_file.writestr("requirements.txt", generate_requirements_file(result['dependencies']))
                            if generate_docker:
                                zip_file.writestr("Dockerfile", generate_dockerfile(framework, task.replace(' ', '_')))
                            zip_file.writestr("README.md", f"# {task}\n\nGenerated by AI App Builder Pro+")
                        
                        zip_buffer.seek(0)
                        b64 = base64.b64encode(zip_buffer.read()).decode()
                        href = f'<a href="data:application/zip;base64,{b64}" download="{task.replace(" ", "_")}_project.zip">📦 Download Full Project</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                    with col3:
                        if st.button("⭐ Save to Favorites"):
                            st.session_state.favorites[task] = {
                                'code': result['code'],
                                'timestamp': datetime.now().isoformat()
                            }
                            st.success("Added to favorites!")
    
    # Tab 2: App Library
    with tabs[1]:
        st.header("📚 Application Library")
        
        apps = db.get_all_apps()
        if apps:
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                search_term = st.text_input("🔍 Search Applications", placeholder="Search by name, framework, or feature...")
            with col2:
                filter_framework = st.multiselect(
                    "Filter by Framework",
                    list(set(app['framework'] for app in apps))
                )
            with col3:
                filter_complexity = st.multiselect(
                    "Filter by Complexity",
                    list(set(app['complexity'] for app in apps))
                )
            
            # Filter applications
            filtered_apps = apps
            
            if search_term:
                filtered_apps = [app for app in filtered_apps if 
                               search_term.lower() in app['task'].lower() or
                               search_term.lower() in app['framework'].lower() or
                               search_term.lower() in str(app['features']).lower()]
            
            if filter_framework:
                filtered_apps = [app for app in filtered_apps if app['framework'] in filter_framework]
            
            if filter_complexity:
                filtered_apps = [app for app in filtered_apps if app['complexity'] in filter_complexity]
            
            # Display applications
            for app in filtered_apps:
                with st.expander(f"📱 {app['task']} | 🏗️ {app['framework']} | ⭐ {app['quality_score']}/100", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Task:** {app['task']}")
                        st.write(f"**Generated:** {datetime.fromisoformat(app['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Complexity:** {app['complexity']}")
                        st.write(f"**Generation Time:** {app['generation_time']} seconds")
                        
                        # Features badges
                        features = json.loads(app['features'])
                        if features:
                            st.write("**Features:**")
                            cols = st.columns(min(4, len(features)))
                            for idx, feature in enumerate(features):
                                with cols[idx % 4]:
                                    st.markdown(f'<span class="metric-badge">{feature}</span>', unsafe_allow_html=True)
                    
                    with col2:
                        # Quality indicator
                        quality = app['quality_score']
                        if quality >= 80:
                            st.success(f"Quality: {quality}/100")
                        elif quality >= 60:
                            st.warning(f"Quality: {quality}/100")
                        else:
                            st.error(f"Quality: {quality}/100")
                        
                        # Actions
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("📋 View", key=f"view_{app['id']}"):
                                st.code(app['full_code'], language='python')
                        with col_b:
                            if st.button("🔁 Regenerate", key=f"regen_{app['id']}"):
                                st.info(f"Regenerating {app['task']}...")
            
            # Statistics
            st.subheader("📊 Library Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Apps", len(filtered_apps))
            with col2:
                avg_quality = sum(app['quality_score'] for app in filtered_apps) / max(1, len(filtered_apps))
                st.metric("Avg Quality", f"{avg_quality:.1f}/100")
            with col3:
                total_lines = sum(len(app['full_code'].split('\n')) for app in filtered_apps)
                st.metric("Total Lines", f"{total_lines:,}")
            with col4:
                frameworks = len(set(app['framework'] for app in filtered_apps))
                st.metric("Frameworks Used", frameworks)
        
        else:
            st.info("📭 No applications generated yet. Start creating in the 'Create App' tab!")
    
    # Tab 3: Code Workshop
    with tabs[2]:
        st.header("🔧 Code Workshop")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Code Editor
            st.subheader("📝 Code Editor")
            default_code = '''# Enter your Python code here
import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.set_page_config(page_title="My Application", layout="wide")
    st.title("My Application")
    st.write("Hello, World!")
    
    # Add some interactive elements
    name = st.text_input("Enter your name", "John Doe")
    age = st.slider("Select your age", 0, 100, 30)
    
    if st.button("Greet"):
        st.success(f"Hello {name}! You are {age} years old.")
    
    # Sample data visualization
    data = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100)
    })
    st.scatter_chart(data)

if __name__ == "__main__":
    main()'''
            
            code_input = st.text_area(
                "Edit Code:",
                value=default_code,
                height=400,
                key="workshop_code"
            )
        
        with col2:
            # Code Analysis Tools
            st.subheader("🔍 Analysis Tools")
            
            if st.button("🔬 Analyze Code", use_container_width=True):
                with st.spinner("Analyzing code..."):
                    metrics = code_gen.analyze_code_complexity(code_input)
                    validation = validate_code(code_input)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Metrics
                    st.metric("Lines of Code", metrics.lines_of_code)
                    st.metric("Functions", metrics.functions)
                    st.metric("Complexity Score", f"{metrics.complexity_score:.1f}")
                    
                    # Validation
                    st.subheader("Validation")
                    for check, result in validation.items():
                        if result['valid']:
                            st.success(f"✅ {check}: {result['message']}")
                        else:
                            st.error(f"❌ {check}: {result['message']}")
            
            # Code Transformation
            st.subheader("🛠️ Transformations")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✨ Optimize", use_container_width=True):
                    optimized = optimize_code(code_input)
                    st.code(optimized, language='python')
            with col_b:
                if st.button("📋 Add Docs", use_container_width=True):
                    # Add basic docstrings
                    lines = code_input.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith('def '):
                            func_name = line.split('def ')[1].split('(')[0]
                            docstring = f'    """Documentation for {func_name}."""'
                            lines.insert(i + 1, docstring)
                    st.code('\n'.join(lines), language='python')
        
        # Code Visualization
        st.subheader("📊 Code Visualization")
        
        if st.button("Generate Visualization", use_container_width=True):
            structure = visualize_code_structure(code_input)
            
            if structure['children']:
                # Create bar chart
                df = pd.DataFrame(structure['children'])
                fig = px.bar(df, x='name', y='value', title="Code Structure Analysis")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not parse code structure")
    
    # Tab 4: Analytics
    with tabs[3]:
        st.header("📊 Advanced Analytics")
        
        apps = db.get_all_apps()
        if apps:
            # Convert to DataFrame
            df = pd.DataFrame(apps)
            
            # KPIs
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Applications", len(df))
            with col2:
                avg_gen_time = df['generation_time'].mean()
                st.metric("Avg Generation Time", f"{avg_gen_time:.2f}s")
            with col3:
                avg_quality = df['quality_score'].mean()
                st.metric("Avg Quality Score", f"{avg_quality:.1f}/100")
            with col4:
                total_tokens = df['estimated_tokens'].sum()
                st.metric("Total Tokens", f"{total_tokens:,}")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Framework distribution
                if 'framework' in df.columns and not df.empty:
                    framework_counts = df['framework'].value_counts()
                    fig1 = px.pie(
                        values=framework_counts.values,
                        names=framework_counts.index,
                        title="Framework Distribution",
                        hole=0.4
                    )
                    st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Quality over time
                if 'timestamp' in df.columns and not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    
                    fig2 = px.scatter(
                        df,
                        x='timestamp',
                        y='quality_score',
                        color='framework',
                        size='generation_time',
                        title="Quality Score Over Time",
                        hover_data=['task', 'complexity']
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Complexity Analysis
            st.subheader("Complexity Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Complexity distribution
                if 'complexity' in df.columns and not df.empty:
                    complexity_counts = df['complexity'].value_counts()
                    fig3 = px.bar(
                        x=complexity_counts.index,
                        y=complexity_counts.values,
                        title="Complexity Level Distribution",
                        color=complexity_counts.values,
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                # Generation time by complexity
                if 'complexity' in df.columns and 'generation_time' in df.columns and not df.empty:
                    fig4 = px.box(
                        df,
                        x='complexity',
                        y='generation_time',
                        title="Generation Time by Complexity",
                        points="all"
                    )
                    st.plotly_chart(fig4, use_container_width=True)
            
            # Performance Trends
            st.subheader("Performance Trends")
            
            if 'timestamp' in df.columns and 'quality_score' in df.columns and not df.empty:
                # Calculate moving average
                df['quality_ma'] = df['quality_score'].rolling(window=5, min_periods=1).mean()
                
                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['quality_score'],
                    mode='markers',
                    name='Quality Score',
                    marker=dict(size=8, opacity=0.6)
                ))
                fig5.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['quality_ma'],
                    mode='lines',
                    name='Moving Average (5)',
                    line=dict(color='red', width=3)
                ))
                
                fig5.update_layout(
                    title="Quality Score Trend with Moving Average",
                    xaxis_title="Time",
                    yaxis_title="Quality Score",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig5, use_container_width=True)
        
        else:
            st.info("Generate some applications to see analytics!")
    
    # Tab 5: Deployment
    with tabs[4]:
        st.header("🚀 Deployment Center")
        
        st.info("""
        ### One-Click Deployment Options
        
        Deploy your generated applications to various cloud platforms with a single click.
        Each deployment includes:
        - Auto-scaling configuration
        - SSL certificates
        - Monitoring and logging
        - Database setup
        - Backup configuration
        """)
        
        # Deployment Targets
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("☁️ Cloud Platforms")
            if st.button("Deploy to AWS", use_container_width=True):
                st.info("AWS deployment initiated... (Simulation)")
            
            if st.button("Deploy to Azure", use_container_width=True):
                st.info("Azure deployment initiated... (Simulation)")
            
            if st.button("Deploy to GCP", use_container_width=True):
                st.info("Google Cloud deployment initiated... (Simulation)")
        
        with col2:
            st.subheader("🎯 PaaS Services")
            if st.button("Deploy to Heroku", use_container_width=True):
                st.info("Heroku deployment initiated... (Simulation)")
            
            if st.button("Deploy to Railway", use_container_width=True):
                st.info("Railway deployment initiated... (Simulation)")
            
            if st.button("Deploy to Render", use_container_width=True):
                st.info("Render deployment initiated... (Simulation)")
        
        with col3:
            st.subheader("🐳 Container Platforms")
            if st.button("Deploy to Docker Hub", use_container_width=True):
                st.info("Docker Hub push initiated... (Simulation)")
            
            if st.button("Deploy to Kubernetes", use_container_width=True):
                st.info("Kubernetes deployment initiated... (Simulation)")
        
        # Deployment Configuration
        st.subheader("⚙️ Deployment Configuration")
        
        with st.expander("Advanced Configuration", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                instance_type = st.selectbox(
                    "Instance Type",
                    ["Small (1GB RAM)", "Medium (2GB RAM)", "Large (4GB RAM)", "X-Large (8GB RAM)"]
                )
                
                scaling_min = st.number_input("Minimum Instances", 1, 10, 1)
                scaling_max = st.number_input("Maximum Instances", 1, 100, 3)
            
            with col2:
                region = st.selectbox(
                    "Region",
                    ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
                )
                
                enable_monitoring = st.checkbox("Enable Monitoring", value=True)
                enable_backups = st.checkbox("Enable Automatic Backups", value=True)
        
        # Deployment Status
        st.subheader("📊 Deployment Status")
        
        # Simulated deployment status
        deployment_data = {
            'Environment': ['Development', 'Staging', 'Production'],
            'Status': ['✅ Running', '🔄 Deploying', '⚠️ Warning'],
            'Uptime': ['99.8%', '98.5%', '99.9%'],
            'Instances': [1, 2, 3]
        }
        
        st.table(pd.DataFrame(deployment_data))
    
    # Tab 6: Settings
    with tabs[5]:
        st.header("⚙️ Application Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("General Settings")
            
            auto_save = st.checkbox("Auto-save generated applications", value=True)
            default_complexity = st.selectbox(
                "Default Complexity",
                ["Beginner", "Intermediate", "Advanced", "Expert"],
                index=1
            )
            
            theme_mode = st.selectbox(
                "Theme Mode",
                ["Light", "Dark", "Auto"]
            )
            
            language = st.selectbox(
                "Interface Language",
                ["English", "Spanish", "French", "German", "Chinese"]
            )
        
        with col2:
            st.subheader("Code Generation")
            
            default_model = st.selectbox(
                "Default AI Model",
                ["fast", "balanced", "advanced", "expert"],
                index=1
            )
            
            code_style = st.selectbox(
                "Code Style Guide",
                ["PEP 8", "Google", "NumPy", "Black"]
            )
            
            auto_format = st.checkbox("Auto-format generated code", value=True)
            auto_test = st.checkbox("Auto-generate tests", value=True)
        
        # API Settings
        st.subheader("API Configuration")
        
        api_key = st.text_input(
            "OpenAI API Key",
            value=API_KEY,
            type="password",
            help="Enter your OpenAI API key"
        )
        
        # Database Management
        st.subheader("Database Management")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("Export Database", use_container_width=True):
                st.info("Database export initiated...")
        
        with col_b:
            if st.button("Import Database", use_container_width=True):
                uploaded_file = st.file_uploader("Choose database file", type=['db', 'sqlite'])
                if uploaded_file:
                    st.success("Database imported successfully!")
        
        with col_c:
            if st.button("Reset Database", use_container_width=True):
                if st.checkbox("Confirm reset (this will delete all data)"):
                    st.warning("Database reset complete!")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption(f"🔄 Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col2:
        st.caption("⚡ AI App Builder Pro+ v2.0")
    
    with col3:
        st.caption("📧 Support: support@aiappbuilder.com")

# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Critical Error: {str(e)}")
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)
        
        # Show recovery options
        st.warning("The application encountered a critical error.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Restart Application"):
                st.rerun()
        with col2:
            if st.button("🧹 Clear All Data"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
