package il.ac.openu.bestparents;

import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.core.ContingencyTables;
import weka.core.Instances;

/**
 * Best parents search.
 *
 * @author Andrew Kreimer
 */
public class BestParentsSearch extends SearchAlgorithm {

  private static final long serialVersionUID = -8315181456697597693L;

  /**
   * Performs path search.
   *
   * @param bayesNet the network
   * @param instances the data to work with
   */
  @Override
  public void search(BayesNet bayesNet, Instances instances) {
    // contingency table for each [attribute X attribute] matrix
    var attributeMatrix = allocateAttributeMatrix(instances);

    // count instantiations
    countInstantiations(instances, attributeMatrix);

    // for each attribute with index i: map<entropy, parent index>, keeping the map sorted
    var attributeBestParentsList = allocateAttributeMaps(instances);

    findBestParents(instances, attributeMatrix, attributeBestParentsList);

    // add good parents, for each attribute, bounded by maxNumberOfParents
    addBestRules(bayesNet, instances, attributeBestParentsList);
  }

  private void addBestRules(
      BayesNet bayesNet,
      Instances instances,
      List<SortedMap<Double, Integer>> attributeBestParentsList) {
    for (var i = 0; i < instances.numAttributes(); i++) {
      var tmpTreeMap = attributeBestParentsList.get(i);
      var numOfAddedRules = 0;

      for (Entry<Double, Integer> entry : tmpTreeMap.entrySet()) {
        int value = entry.getValue();

        if (numOfAddedRules < getMaxNrOfParents()
            && numOfAddedRules < tmpTreeMap.size()
            &&
            // avoid parents with several children
            BnUtils.countNumOfChildren(bayesNet, instances, value) < getMaxNrOfParents()
            && !bayesNet.getParentSet(i).contains(value)) {
          bayesNet.getParentSet(i).addParent(value, instances);
          numOfAddedRules++;
        }
      }
    }
  }

  /** Finds the best parents for each attribute by conditional entropy and greedy algorithm. */
  private void findBestParents(
      Instances instances,
      double[][][][] attributeMatrix,
      List<SortedMap<Double, Integer>> attributeBestParentsList) {
    // calculate conditional entropy for contingency tables
    for (var i = 0; i < instances.numAttributes(); i++) {
      for (var j = 0; j < i; j++) {
        var entropyConditionedOnRows =
            ContingencyTables.entropyConditionedOnRows(attributeMatrix[i][j]);
        var entropyConditionedOnColumns =
            ContingencyTables.entropyConditionedOnColumns(attributeMatrix[i][j]);

        var lowestEntropy =
            (entropyConditionedOnRows < entropyConditionedOnColumns)
                ? entropyConditionedOnRows
                : entropyConditionedOnColumns;

        // best rule
        if (entropyConditionedOnRows < entropyConditionedOnColumns) {
          attributeBestParentsList.get(j).put(lowestEntropy, i);
        } else {
          attributeBestParentsList.get(i).put(lowestEntropy, j);
        }
      }
    }
  }

  /** Allocates List of Map: Map of best parents for each attribute. */
  private List<SortedMap<Double, Integer>> allocateAttributeMaps(Instances instances) {
    var attributeBestParentsList = new ArrayList<SortedMap<Double, Integer>>();

    // allocate
    for (var i = 0; i < instances.numAttributes(); i++) {
      var tmpTreeMap = new TreeMap<Double, Integer>();
      attributeBestParentsList.add(i, tmpTreeMap);
    }

    return attributeBestParentsList;
  }

  /** Counts instances for each attribute and category. */
  private void countInstantiations(Instances instances, double[][][][] attributeMatrix) {
    for (var n = 0; n < instances.numInstances(); n++) {
      for (var i = 0; i < instances.numAttributes(); i++) {
        for (var j = 0; j < i; j++) {
          var iAttrIndex = (int) instances.instance(n).value(i);
          var jAttrIndex = (int) instances.instance(n).value(j);
          attributeMatrix[i][j][iAttrIndex][jAttrIndex]++;
        }
      }
    }
  }

  /** Allocates contingency matrix for each attribute-attribute pair. */
  private double[][][][] allocateAttributeMatrix(Instances instances) {
    var attributeMatrix = new double[instances.numAttributes()][instances.numAttributes()][][];

    // allocate
    for (var j = 0; j < instances.numAttributes(); j++) {
      for (var k = 0; k < j; k++) {
        attributeMatrix[j][k] =
            new double[instances.attribute(j).numValues()][instances.attribute(k).numValues()];
      }
    }

    return attributeMatrix;
  }

  /** Sets the max number of parents. */
  public void setMaxNrOfParents(int nMaxNrOfParents) {
    m_nMaxNrOfParents = nMaxNrOfParents;
  }

  /** Gets the max number of parents. */
  public int getMaxNrOfParents() {
    return m_nMaxNrOfParents;
  }
}
