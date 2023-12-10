import requests
import xml.etree.ElementTree as ET
import csv

# URL of the XML file hosted on the website
url = 'https://coagmet.colostate.edu/data/metadata.xml?units=m'

# Fetch the XML content from the website
response = requests.get(url)

if response.status_code == 200:
    # Parse XML content
    root = ET.fromstring(response.content)

    # Open a CSV file in write mode
    with open('data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        # Create a CSV writer object
        csvwriter = csv.writer(csvfile)

        # Write headers to the CSV file based on XML tags
        headers = [child.tag for child in root[0]]
        csvwriter.writerow(headers)

        # Write data to the CSV file
        for record in root:
            data = [child.text for child in record]
            csvwriter.writerow(data)

    print("CSV file has been created successfully.")
else:
    print(f"Failed to fetch data from {url}. Status code: {response.status_code}")
