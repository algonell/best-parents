package il.ac.openu.bestparents;

import java.util.AbstractMap;
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
 * Best children search.
 *
 * @author Andrew Kreimer
 */
public class BestChildrenSearch extends SearchAlgorithm {

  private static final long serialVersionUID = 1032285588625105530L;

  private int maxNrOfChildren;
  private double[][][][] attributeMatrix = {};

  /**
   * Performs path search.
   *
   * @param bayesNet the network
   * @param instances the data to work with
   */
  @Override
  public void search(BayesNet bayesNet, Instances instances) {
    // contingency table for each attribute X attribute matrix
    attributeMatrix = new double[instances.numAttributes()][instances.numAttributes()][][];

    // allocate
    allocate(instances);

    // count instantiations
    count(instances);

    // for each attribute with index i: map<entropy, child index>, keeping the map sorted
    var attributeBestChildrenList = new ArrayList<SortedMap<Double, Integer>>();

    // allocate
    for (var i = 0; i < instances.numAttributes(); i++) {
      var tmpTreeMap = new TreeMap<Double, Integer>();
      attributeBestChildrenList.add(i, tmpTreeMap);
    }

    // map<entropy, rule(string)>
    var entropyRuleMap = new TreeMap<Double, String>();

    // map<entropy, rule(attributeParentIndex -> attributeChildIndex)>
    var entropyParentToChildMap = new TreeMap<Double, Entry<Integer, Integer>>();

    // calculate conditional entropy for contingency tables
    calculateContingencyTables(
        instances, attributeBestChildrenList, entropyRuleMap, entropyParentToChildMap);

    // build network
    assembleNetwork(bayesNet, instances, attributeBestChildrenList);
  }

  /** Assembles network. */
  private void assembleNetwork(
      BayesNet bayesNet,
      Instances instances,
      List<SortedMap<Double, Integer>> attributeBestChildrenList) {
    for (var i = 0; i < instances.numAttributes(); i++) {
      var tmpTreeMap = attributeBestChildrenList.get(i);
      var numOfAddedRules = 0;

      for (Entry<Double, Integer> entry : tmpTreeMap.entrySet()) {
        int value = entry.getValue();

        var numOfParentsForCurrentChild = bayesNet.getParentSet(value).getNrOfParents();
        if (numOfAddedRules < getMaxNrOfChildren()
            && numOfParentsForCurrentChild < getMaxNrOfChildren()
            && numOfAddedRules < tmpTreeMap.size()
            && BnUtils.countNumOfChildren(bayesNet, instances, i) < getMaxNrOfChildren()
            && !bayesNet.getParentSet(value).contains(i)) {
          bayesNet.getParentSet(value).addParent(i, instances);
          numOfAddedRules++;
        }
      }
    }
  }

  /** Calculates conditional entropies. */
  private void calculateContingencyTables(
      Instances instances,
      List<SortedMap<Double, Integer>> attributeBestChildrenList,
      SortedMap<Double, String> entropyRuleMap,
      SortedMap<Double, Entry<Integer, Integer>> entropyParentToChildMap) {
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

        // save current rule
        var arc =
            (entropyConditionedOnRows < entropyConditionedOnColumns)
                ? instances.attribute(i).name() + " -> " + instances.attribute(j).name()
                : instances.attribute(j).name() + " -> " + instances.attribute(i).name();
        entropyRuleMap.put(lowestEntropy, arc);

        if (entropyConditionedOnRows < entropyConditionedOnColumns) {
          attributeBestChildrenList.get(i).put(lowestEntropy, j);
          entropyParentToChildMap.put(lowestEntropy, new AbstractMap.SimpleEntry<>(i, j));
        } else {
          attributeBestChildrenList.get(j).put(lowestEntropy, i);
          entropyParentToChildMap.put(lowestEntropy, new AbstractMap.SimpleEntry<>(j, i));
        }
      }
    }
  }

  /** Counts occurrences. */
  private void count(Instances instances) {
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

  /** Allocates memory. */
  private void allocate(Instances instances) {
    for (var j = 0; j < instances.numAttributes(); j++) {
      for (var k = 0; k < j; k++) {
        attributeMatrix[j][k] =
            new double[instances.attribute(j).numValues()][instances.attribute(k).numValues()];
      }
    }
  }

  /** Sets the max number of children. */
  public void setMaxNrOfChildren(int nMaxNrOfChildren) {
    maxNrOfChildren = nMaxNrOfChildren;
  }

  /** Gets the max number of children. */
  public int getMaxNrOfChildren() {
    return maxNrOfChildren;
  }
}
