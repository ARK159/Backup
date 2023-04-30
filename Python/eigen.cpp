#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

// Node structure for Decision Tree
struct Node {
    string attribute; // Name of the attribute
    vector<Node*> children; // Children of the node
    string label; // Class label
};

// Read CSV file and return dataset
vector<vector<string> > readCsvFile(string fileName) {
    vector<vector<string> > dataset;
    ifstream file(fileName.c_str());
    string line;

    while (getline(file, line)) {
        vector<string> row;
        stringstream ss(line);
        string value;

        while (getline(ss, value, ',')) {
            row.push_back(value);
        }

        dataset.push_back(row);
    }

    file.close();

    return dataset;
}

// Split dataset into training and validation sets
void splitDataset(vector<vector<string> > dataset, float trainingRatio, vector<vector<string> >& trainingSet, vector<vector<string> >& validationSet) {
    int size = dataset.size();
    int trainingSize = round(size * trainingRatio);

    vector<int> indices(size);
    for (int i = 0; i < size; i++) {
        indices[i] = i;
    }

    srand(time(NULL));
    random_shuffle(indices.begin(), indices.end());

    for (int i = 0; i < size; i++) {
        if (i < trainingSize) {
            trainingSet.push_back(dataset[indices[i]]);
        } else {
            validationSet.push_back(dataset[indices[i]]);
        }
    }
}

// Calculate entropy of dataset
double calculateEntropy(vector<vector<string> > dataset) {
    map<string, int> classCounts;
    double entropy = 0.0;

    for (int i = 0; i < dataset.size(); i++) {
        string label = dataset[i][dataset[0].size() - 1];
        classCounts[label]++;
    }

    for (auto it = classCounts.begin(); it != classCounts.end(); it++) {
        double probability = (double)it->second / dataset.size();
        entropy -= probability * log2(probability);
    }

    return entropy;
}

// Split dataset based on attribute value
vector<vector<vector<string> > > splitDatasetByAttribute(vector<vector<string> > dataset, string attribute) {
    vector<vector<vector<string> > > subsets(dataset[0].size() - 1);

    for (int i = 0; i < dataset.size(); i++) {
        for (int j = 0; j < dataset[0].size() - 1; j++) {
            if (dataset[0][j] == attribute) {
                subsets[j][stoi(dataset[i][j])].push_back(dataset[i]);
                break;
            }
        }
    }

    return subsets;
}

// Calculate information gain of attribute
double calculateInformationGain(vector<vector<string> > dataset, string attribute) {
    double informationGain = calculateEntropy(dataset);
    vector<vector<vector<string> > > subsets = splitDatasetByAttribute(dataset, attribute);

    for (int i = 0; i < subsets.size(); i++) {
        double probability = (double)subsets[i][0].size() / dataset.size();
        double entropy = calculateEntropy(subsets[i][0]);
        informationGain -= probability * entropy;
    }

    return informationGain;
}

// Find attribute with maximum information gain
string findBestAttribute(vector<vector<string> >(string attribute, vector<string> remainingAttributes, vector<vector<string> > dataset) {
double maxInformationGain = -1.0;
string bestAttribute;
for (int i = 0; i < remainingAttributes.size(); i++) {
    double informationGain = calculateInformationGain(dataset, remainingAttributes[i]);

    if (informationGain > maxInformationGain) {
        maxInformationGain = informationGain;
        bestAttribute = remainingAttributes[i];
    }
}

if (maxInformationGain <= 0.0) {
    return attribute;
}

return bestAttribute;
}

// Build decision tree using ID3 algorithm
Node* buildDecisionTree(vector<vector<string> > dataset, vector<string> attributes) {
Node* node = new Node;
string firstLabel = dataset[0][dataset[0].size() - 1];
bool sameClass = true;

for (int i = 1; i < dataset.size(); i++) {
    if (dataset[i][dataset[0].size() - 1] != firstLabel) {
        sameClass = false;
        break;
    }
}

if (sameClass) {
    node->label = firstLabel;
    return node;
}

// Check if there are no more attributes to split on
if (attributes.empty()) {
    node->label = firstLabel;
    return node;
}

// Find attribute with maximum information gain
string bestAttribute = findBestAttribute(firstLabel, attributes, dataset);
node->attribute = bestAttribute;
vector<string> remainingAttributes = attributes;

for (int i = 0; i < remainingAttributes.size(); i++) {
    if (remainingAttributes[i] == bestAttribute) {
        remainingAttributes.erase(remainingAttributes.begin() + i);
        break;
    }
}

// Split dataset based on best attribute
vector<vector<vector<string> > > subsets = splitDatasetByAttribute(dataset, bestAttribute);

for (int i = 0; i < subsets[0].size(); i++) {
    Node* child = new Node;

    if (subsets[0][i].empty()) {
        child->label = firstLabel;
    } else {
        child = buildDecisionTree(subsets[0][i], remainingAttributes);
    }

    node->children.push_back(child);
}

return node;
}

// Predict class label using decision tree
string predictClassLabel(Node* root, vector<string> example) {
Node* node = root;
while (!node->children.empty()) {
    string attribute = node->attribute;

    for (int i = 0; i < example.size() - 1; i++) {
        if (attribute == example[i]) {
            node = node->children[stoi(example[i])];
            break;
        }
    }
}

return node->label;
}

// Train Random Forest algorithm
vector<Node*> trainRandomForest(vector<vector<string> > dataset, int numTrees, float trainingRatio) {
vector<Node*> forest;
for (int i = 0; i < numTrees; i++) {
    vector<vector<string> > trainingSet;
    vector<vector<string> > validationSet;
    vector<string> attributes = dataset[0];

    splitDataset(dataset, trainingRatio, trainingSet, validationSet);

    Node* root = buildDecisionTree(trainingSet, attributes);
    forest.push_back(root);
}

return forest;
}

// Predict class label using Random Forest algorithm
string predictClassLabelRandomForest(vector<Node*> forest, vector<string> example) {
map<string, int> classCounts;
for (int i =0; i < forest.size(); i++) {
string predictedLabel = predictClassLabel(forest[i], example);
classCounts[predictedLabel]++;
}
int maxCount = -1;
string predictedLabel;

for (auto it = classCounts.begin(); it != classCounts.end(); it++) {
    if (it->second > maxCount) {
        maxCount = it->second;
        predictedLabel = it->first;
    }
}

return predictedLabel;
// Test Random Forest algorithm
void testRandomForest(vector<Node*> forest, vector<vector<string> > testSet) {
int numCorrectPredictions = 0;
for (int i = 0; i < testSet.size(); i++) {
    string predictedLabel = predictClassLabelRandomForest(forest, testSet[i]);

    if (predictedLabel == testSet[i][testSet[i].size() - 1]) {
        numCorrectPredictions++;
    }
}

double accuracy = (double) numCorrectPredictions / (double) testSet.size();
cout << "Random Forest accuracy: " << accuracy << endl;
int main() {
vector<vector<string> > dataset = readCsv("iris.csv");
random_shuffle(dataset.begin(), dataset.end());

// Split dataset into training and test sets
vector<vector<string> > trainingSet;
vector<vector<string> > testSet;
splitDataset(dataset, 0.8, trainingSet, testSet);

// Train Random Forest algorithm
vector<Node*> forest = trainRandomForest(trainingSet, 10, 0.8);

// Test Random Forest algorithm
testRandomForest(forest, testSet);

return 0;
}