import requests
import pandas as pd
import json

def search_socrata_datasets():
    """
    Search for datasets using the working Socrata API endpoints
    """
    print("🔍 SEARCHING FOR OVERSEAS FREIGHT DATASETS...")
    
    # Try different search approaches
    search_terms = [
        "shipping",
        "maritime", 
        "international trade",
        "port cargo",
        "container",
        "freight",
        "overseas",
        "import export"
    ]
    
    for term in search_terms:
        print(f"\n📋 Searching for '{term}'...")
        
        # Try the Open Data Network search
        search_url = f"https://api.us.socrata.com/api/catalog/v1/datasets?query={term}&limit=10"
        
        try:
            response = requests.get(search_url, timeout=10)
            if response.status_code == 200:
                results = response.json()
                datasets = results.get('results', [])
                print(f"   Found {len(datasets)} datasets")
                
                for dataset in datasets[:3]:  # Show top 3
                    resource = dataset.get('resource', {})
                    name = resource.get('name', 'Unknown')
                    domain = resource.get('domain', 'Unknown')
                    print(f"     - {name}")
                    print(f"       Domain: {domain}")
            else:
                print(f"   ❌ HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error: {e}")

def find_government_datasets():
    """
    Search for government datasets that might contain overseas freight data
    """
    print("\n🏛️ SEARCHING GOVERNMENT DATASETS...")
    
    # Common government domains that might have freight data
    government_domains = [
        "data.gov",
        "data.commerce.gov", 
        "data.transportation.gov",
        "data.census.gov",
        "data.usa.gov"
    ]
    
    for domain in government_domains:
        print(f"\n📋 Checking {domain}...")
        
        # Try to find datasets on this domain
        search_url = f"https://{domain}/api/catalog/v1/datasets?query=shipping&limit=5"
        
        try:
            response = requests.get(search_url, timeout=10)
            if response.status_code == 200:
                results = response.json()
                datasets = results.get('results', [])
                print(f"   Found {len(datasets)} shipping-related datasets")
                
                for dataset in datasets:
                    resource = dataset.get('resource', {})
                    name = resource.get('name', 'Unknown')
                    print(f"     - {name}")
            else:
                print(f"   ❌ HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error: {e}")

def search_specific_datasets():
    """
    Search for specific known overseas freight datasets
    """
    print("\n🎯 SEARCHING SPECIFIC DATASETS...")
    
    # Known overseas freight datasets (these are examples)
    known_datasets = [
        {
            "name": "Port of Los Angeles Cargo Statistics",
            "url": "https://data.lacity.org/resource/",
            "domain": "data.lacity.org"
        },
        {
            "name": "Port of Long Beach Cargo Data", 
            "url": "https://data.longbeach.gov/resource/",
            "domain": "data.longbeach.gov"
        },
        {
            "name": "US Census Bureau Trade Data",
            "url": "https://data.census.gov/resource/",
            "domain": "data.census.gov"
        }
    ]
    
    for dataset in known_datasets:
        print(f"\n📋 Checking {dataset['name']}...")
        print(f"   Domain: {dataset['domain']}")
        print(f"   URL: {dataset['url']}")
        
        # Try to access the dataset
        try:
            response = requests.get(f"{dataset['url']}.json", timeout=10)
            if response.status_code == 200:
                print(f"   ✅ Dataset accessible")
                data = response.json()
                if data:
                    print(f"   📊 Found {len(data)} records")
                    # Show sample columns if available
                    if data and len(data) > 0:
                        sample = data[0]
                        columns = list(sample.keys())
                        print(f"   📋 Sample columns: {columns[:5]}...")
            else:
                print(f"   ❌ HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error: {e}")

def search_open_data_network():
    """
    Search the Open Data Network for overseas freight data
    """
    print("\n🌐 SEARCHING OPEN DATA NETWORK...")
    
    # Try the Open Data Network search
    search_url = "https://api.us.socrata.com/api/catalog/v1/datasets"
    
    search_params = {
        "query": "overseas freight international shipping",
        "limit": 20,
        "offset": 0
    }
    
    try:
        response = requests.get(search_url, params=search_params, timeout=15)
        if response.status_code == 200:
            results = response.json()
            datasets = results.get('results', [])
            print(f"   Found {len(datasets)} overseas freight datasets")
            
            for dataset in datasets[:5]:
                resource = dataset.get('resource', {})
                name = resource.get('name', 'Unknown')
                domain = resource.get('domain', 'Unknown')
                description = resource.get('description', 'No description')
                print(f"\n     📋 {name}")
                print(f"        Domain: {domain}")
                print(f"        Description: {description[:100]}...")
        else:
            print(f"   ❌ HTTP {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error: {e}")

def main():
    """
    Main function to search for overseas freight data
    """
    print("🚢 OVERSEAS FREIGHT DATA SEARCH")
    print("=" * 50)
    
    # Try multiple search approaches
    search_socrata_datasets()
    find_government_datasets() 
    search_specific_datasets()
    search_open_data_network()
    
    print("\n✅ SEARCH COMPLETE!")
    print("\n📋 ALTERNATIVE APPROACHES:")
    print("   1. Visit https://data.gov and search for 'shipping' or 'maritime'")
    print("   2. Check https://www.census.gov/data/datasets.html for trade data")
    print("   3. Visit https://www.bts.gov/data for transportation data")
    print("   4. Check https://www.commerce.gov/data for trade statistics")
    print("   5. Use https://www.rita.dot.gov/bts/sites/rita.dot.gov.bts/files/publications/")
    print("      for Bureau of Transportation Statistics data")

if __name__ == "__main__":
    main() 