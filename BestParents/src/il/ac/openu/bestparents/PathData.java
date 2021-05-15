/*******************************************************************************
 * Copyright (c) Algonell.com
 * Algonell Confidential
 * All Rights Reserved
 ******************************************************************************/
package il.ac.openu.bestparents;

import java.util.SortedMap;

import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;

/**
 * Auto generated wrapper for path expansion.
 * 
 * @author Andrew Kreimer
 *
 */
public class PathData {
	
	private BayesNet bayesNet;
	private Instances instances;
	private int i;
	private boolean[] blackList;
	private SortedMap<Double, Integer> tmpBestChildrenMap;
	private SortedMap<Double, Integer> tmpBestParentsMap;
	private double bestChildKey;
	private double bestParentKey;
	private Integer bestChild;
	private Integer bestParent;
	private boolean expandToChild;
	private boolean expandToParent;

	public BayesNet getBayesNet() { return bayesNet; }
	public void setBayesNet(BayesNet bayesNet) { this.bayesNet = bayesNet; }
	public Instances getInstances() { return instances; }
	public void setInstances(Instances instances) { this.instances = instances; }
	public int getI() { return i; }
	public void setI(int i) { this.i = i; }
	public boolean[] getBlackList() { return blackList; }
	public void setBlackList(boolean[] blackList) { this.blackList = blackList; }
	public SortedMap<Double, Integer> getTmpBestChildrenMap() { return tmpBestChildrenMap; }
	public void setTmpBestChildrenMap(SortedMap<Double, Integer> tmpBestChildrenMap) { 
		this.tmpBestChildrenMap = tmpBestChildrenMap; }
	public SortedMap<Double, Integer> getTmpBestParentsMap() { return tmpBestParentsMap; }
	public void setTmpBestParentsMap(SortedMap<Double, Integer> tmpBestParentsMap) { 
		this.tmpBestParentsMap = tmpBestParentsMap; }
	public double getBestChildKey() { return bestChildKey; }
	public void setBestChildKey(double bestChildKey) { this.bestChildKey = bestChildKey; }
	public double getBestParentKey() { return bestParentKey; }
	public void setBestParentKey(double bestParentKey) { this.bestParentKey = bestParentKey; }
	public Integer getBestChild() { return bestChild; }
	public void setBestChild(Integer bestChild) { this.bestChild = bestChild; }
	public Integer getBestParent() { return bestParent; }
	public void setBestParent(Integer bestParent) { this.bestParent = bestParent; }
	public boolean isExpandToChild() { return expandToChild; }
	public void setExpandToChild(boolean expandToChild) { this.expandToChild = expandToChild; }
	public boolean isExpandToParent() { return expandToParent; }
	public void setExpandToParent(boolean expandToParent) { this.expandToParent = expandToParent; }
	
}