from flask import Flask, render_template_string, request
import joblib
from urllib.parse import urlparse
import whois
import re
from datetime import datetime
import numpy as np
import os

app = Flask(__name__)

# Model loading check: only production_model.pkl is now required.
if not os.path.exists('production_model.pkl'):
    raise FileNotFoundError("Missing production_model.pkl")

class PhishingDetector:
    def __init__(self):
        self.model = joblib.load('production_model.pkl')
        
    def analyze_url(self, url):
        parsed = urlparse(url)
        if not parsed.netloc:
            raise ValueError("Invalid URL")
        
        # Extended feature extraction to compute 16 features
        url_length = len(url)
        num_digits = sum(c.isdigit() for c in url)
        num_params = len(parsed.query.split('&')) if parsed.query else 0
        has_https = 1 if parsed.scheme == 'https' else 0
        num_subdomains = parsed.netloc.count('.')
        domain_age = self.get_domain_age(parsed.netloc)
        is_ip = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", parsed.netloc) else 0
        num_hyphens = url.count('-')
        num_underscores = url.count('_')
        num_at = url.count('@')
        num_slashes = url.count('/') - 2  # subtract protocol slashes
        port = parsed.port if parsed.port else None
        non_std_port = 1 if port and port not in [80, 443] else 0
        domain_length = len(parsed.netloc)
        path_length = len(parsed.path)
        num_letters = sum(c.isalpha() for c in url)
        letter_digit_ratio = num_letters / (num_digits + 1)  # avoid division by zero
        
        from math import log2
        from collections import Counter
        counts = Counter(url)
        entropy = -sum((count / len(url)) * log2(count / len(url)) for count in counts.values())
        
        features = {
            'url_length': url_length,
            'num_digits': num_digits,
            'num_params': num_params,
            'has_https': has_https,
            'num_subdomains': num_subdomains,
            'domain_age': domain_age,
            'is_ip': is_ip,
            'num_hyphens': num_hyphens,
            'num_underscores': num_underscores,
            'num_at': num_at,
            'num_slashes': num_slashes,
            'non_std_port': non_std_port,
            'domain_length': domain_length,
            'path_length': path_length,
            'letter_digit_ratio': letter_digit_ratio,
            'entropy': entropy
        }
        
        # Ensure features are in the correct order expected by the model
        feature_order = ['url_length', 'num_digits', 'num_params', 'has_https', 'num_subdomains',
                         'domain_age', 'is_ip', 'num_hyphens', 'num_underscores', 'num_at',
                         'num_slashes', 'non_std_port', 'domain_length', 'path_length',
                         'letter_digit_ratio', 'entropy']
        feature_list = [features[key] for key in feature_order]
        
        # Directly use the raw features without scaling
        features_arr = np.array([feature_list])
        prediction = bool(self.model.predict(features_arr)[0])
        return prediction, features

    def get_domain_age(self, domain):
        try:
            info = whois.whois(domain)
            created = info.creation_date
            if isinstance(created, list):
                created = created[0]
            return (datetime.now() - created).days if created else 0
        except:
            return 0

detector = PhishingDetector()

IOS_UI = '''
<!DOCTYPE html>
<html>
<head>
    <title>PhishGuard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script>
    document.addEventListener('DOMContentLoaded', () => {
        // iOS-style button interaction
        document.querySelectorAll('button').forEach(btn => {
            btn.addEventListener('touchstart', () => btn.classList.add('active'));
            btn.addEventListener('touchend', () => btn.classList.remove('active'));
        });
    });
    </script>
    <style>
        :root {
            --ios-blue: #007AFF;
            --ios-red: #FF3B30;
            --ios-green: #34C759;
            --bg-color: #F2F2F7;
            --card-bg: #FFFFFF;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell;
        }
        
        body {
            background: var(--bg-color);
            min-height: 100vh;
        }
        
        .container {
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: var(--card-bg);
            padding: 15px;
            text-align: center;
            border-bottom: 1px solid #E5E5EA;
        }
        
        .scan-card {
            background: var(--card-bg);
            border-radius: 14px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        
        input[type="text"] {
            width: 100%;
            padding: 12px 16px;
            font-size: 17px;
            border: 1px solid #C6C6C8;
            border-radius: 10px;
            margin: 10px 0;
            -webkit-appearance: none;
        }
        
        button {
            background: var(--ios-blue);
            color: white;
            border: none;
            padding: 14px;
            width: 100%;
            border-radius: 10px;
            font-size: 17px;
            font-weight: 500;
            transition: opacity 0.2s;
        }
        
        button.active {
            opacity: 0.7;
        }
        
        .result-card {
            background: var(--card-bg);
            border-radius: 14px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .safe { color: var(--ios-green); }
        .risky { color: var(--ios-red); }
        
        .security-list {
            list-style: none;
            margin: 15px 0;
        }
        
        .security-list li {
            padding: 12px 0;
            border-bottom: 1px solid #F2F2F7;
            display: flex;
            justify-content: space-between;
        }
        
        .error {
            color: var(--ios-red);
            padding: 10px;
            text-align: center;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>PhishGuard</h1>
    </div>
    
    <div class="container">
        <form method="POST" action="/scan" class="scan-card">
            <input type="text" name="url" placeholder="Enter URL" required>
            <button type="submit">Scan URL</button>
        </form>

        <button id="theme-toggle" style="margin-bottom: 10px;">Toggle Light/Dark Mode</button>

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}

        {% if result %}
        <div class="result-card">
            <h2 class="{{ 'safe' if result.is_safe else 'risky' }}">
                {% if result.is_safe %}
                Safe to Use
                {% else %}
                Potential Phishing Risk
                {% endif %}
            </h2>
            
            <p style="color: #86868B; margin: 15px 0;">{{ result.url }}</p>
            
            <ul class="security-list">
                <li>
                    <span>HTTPS Secure</span>
                    <span>{{ '✓' if result.details.has_https else '✗' }}</span>
                </li>
                <li>
                    <span>Domain Age</span>
                    <span>{{ result.details.domain_age }} days</span>
                </li>
                <li>
                    <span>IP Address</span>
                    <span>{{ '✓' if result.details.is_ip else '✗' }}</span>
                </li>
            </ul>
        </div>
        {% endif %}
    </div>
    
    <script>
        // Theme toggle logic
        const toggleButton = document.getElementById('theme-toggle');
        toggleButton.addEventListener('click', () => {
            // Toggle a class on the body to switch between light and dark mode
            document.body.classList.toggle('dark-mode');
        });
    </script>
    
    <style>
        /* Dark mode styles */
        .dark-mode {
            --bg-color: #1c1c1e;
            --card-bg: #2c2c2e;
        }
        
        .dark-mode body {
            background: var(--bg-color);
        }
    </style>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def home():
    return render_template_string(IOS_UI)

@app.route('/scan', methods=['POST'])
def scan():
    url = request.form.get('url', '').strip()
    if not url:
        return render_template_string(IOS_UI, error="Please enter a URL")
    
    try:
        if not url.startswith(('http://', 'https://')):
            url = f'http://{url}'
        
        prediction, features = detector.analyze_url(url)
        return render_template_string(IOS_UI, 
            result={
                'url': url,
                'is_safe': prediction,
                'details': features
            }
        )
    except Exception as e:
        return render_template_string(IOS_UI, error=f"Scan error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)