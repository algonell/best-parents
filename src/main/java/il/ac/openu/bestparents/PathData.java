package il.ac.openu.bestparents;

import java.util.SortedMap;

/**
 * Auto generated wrapper for path expansion.
 *
 * @author Andrew Kreimer
 */
public class PathData {

  private int i;
  private SortedMap<Double, Integer> tmpBestChildrenMap;
  private SortedMap<Double, Integer> tmpBestParentsMap;
  private double bestChildKey;
  private double bestParentKey;
  private Integer bestChild;
  private Integer bestParent;
  private boolean expandToChild;
  private boolean expandToParent;

  public int getI() {
    return i;
  }

  public void setI(int i) {
    this.i = i;
  }

  public SortedMap<Double, Integer> getTmpBestChildrenMap() {
    return tmpBestChildrenMap;
  }

  public void setTmpBestChildrenMap(SortedMap<Double, Integer> tmpBestChildrenMap) {
    this.tmpBestChildrenMap = tmpBestChildrenMap;
  }

  public SortedMap<Double, Integer> getTmpBestParentsMap() {
    return tmpBestParentsMap;
  }

  public void setTmpBestParentsMap(SortedMap<Double, Integer> tmpBestParentsMap) {
    this.tmpBestParentsMap = tmpBestParentsMap;
  }

  public double getBestChildKey() {
    return bestChildKey;
  }

  public void setBestChildKey(double bestChildKey) {
    this.bestChildKey = bestChildKey;
  }

  public double getBestParentKey() {
    return bestParentKey;
  }

  public void setBestParentKey(double bestParentKey) {
    this.bestParentKey = bestParentKey;
  }

  public Integer getBestChild() {
    return bestChild;
  }

  public void setBestChild(Integer bestChild) {
    this.bestChild = bestChild;
  }

  public Integer getBestParent() {
    return bestParent;
  }

  public void setBestParent(Integer bestParent) {
    this.bestParent = bestParent;
  }

  public boolean isExpandToChild() {
    return expandToChild;
  }

  public void setExpandToChild(boolean expandToChild) {
    this.expandToChild = expandToChild;
  }

  public boolean isExpandToParent() {
    return expandToParent;
  }

  public void setExpandToParent(boolean expandToParent) {
    this.expandToParent = expandToParent;
  }
}
