import re

# List of ranges to exclude
excluded_ranges = {":ItemList"}

# Load the RDF-like file
with open('/Volumes/DATI/GitHub/KGRuleLearning/dataset/schemadotorg/schema.ttl', 'r', encoding='utf-8') as f:
    content = f.read()

# --- Extract predicates with domainIncludes and rangeIncludes ---
entries = re.findall(
    r'(:\w+)\s+a\s+rdf:Property\s*;(.*?)(?=\n\s*:\w+\s+a\s+rdf:Property\s*;|\Z)',
    content, re.DOTALL
)
all_properties=[]
for prop, body in entries:
    body_clean = re.sub(r'\s+', ' ', body)
    prop = prop.lstrip(":")

    all_properties.append(prop)

    # DomainIncludes
    domain_matches = re.findall(r':domainIncludes\s+((?::\w+(?:\s*,\s*:\w+)*))', body_clean)
    for match in domain_matches:
        domains = [d.strip().lstrip(":") for d in match.split(',')]
        for d in domains:
            print(f"{prop} domain {d}")

    # RangeIncludes
    range_matches = re.findall(r':rangeIncludes\s+((?::\w+(?:\s*,\s*:\w+)*))', body_clean)
    for match in range_matches:
        ranges = [r.strip().lstrip(":") for r in match.split(',')]
        for r in ranges:
            if f":{r}" not in excluded_ranges:
                print(f"{prop} range {r}")

# --- Extract subClassOf relationships ---
class_entries = re.findall(
    r'(:\w+)\s+a\s+rdfs:Class\s*;(.*?)(?=\n\s*:\w+\s+a\s+rdfs:Class\s*;|\Z)',
    content, re.DOTALL
)

for cls, body in class_entries:
    cls = cls.lstrip(":")
    body_clean = re.sub(r'\s+', ' ', body)
    subclass_match = re.search(r'rdfs:subClassOf\s+:(\w+)', body_clean)
    if subclass_match:
        superclass = subclass_match.group(1)
        print(f"{cls} subClass {superclass}")


#print()
#with open('predicates.txt', 'w') as f:
 #   for line in all_properties:
  #      f.write(f"{line}\n")

