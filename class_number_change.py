import os
import glob
import xml.etree.ElementTree as ET

# Path to the directory containing XML annotation files
xml_dir = 'E:\\ML course\\datasetttttttttt\\xml'

# Desired common class label
common_class = 'object2'

# Iterate through each XML file in the directory
for xml_file in glob.glob(os.path.join(xml_dir, '*.xml')):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the file being processed
        print(f"Processing file: {xml_file}")

        # Check if there are any <object> tags
        objects = root.findall('object')
        if not objects:
            print(f"No <object> tags found in {xml_file}")
            continue

        # Update all <name> tags to the common class label
        for obj in objects:
            name_tag = obj.find('name')
            if name_tag is not None:
                print(f"Updating <name> tag from {name_tag.text} to {common_class}")
                name_tag.text = common_class
            else:
                print(f"No <name> tag found in <object> in {xml_file}")

        # Save the updated XML file
        tree.write(xml_file)
        print(f"File saved: {xml_file}")

    except ET.ParseError as e:
        print(f"Error parsing {xml_file}: {e}")
    except Exception as e:
        print(f"An error occurred with {xml_file}: {e}")








