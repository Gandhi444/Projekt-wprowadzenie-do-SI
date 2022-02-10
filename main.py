import Data_sets
import xml.etree.ElementTree as ET
ConfigTree = ET.parse('Config.xml')
ConfigRoot = ConfigTree.getroot()
if ConfigRoot[0].text=='True':
    Data_sets.clear_train_test()
