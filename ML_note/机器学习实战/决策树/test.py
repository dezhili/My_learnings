#  使用决策树预测隐形眼镜类型
#  ID3算法无法直接处理数值型数据


if __name__ == '__main__':

	import trees
	import treeplotter
	fr = open('lenses.txt')
	lenses = [inst.strip().split('\t') for inst in fr.readlines()]
	lensesLabels = ['sge', 'prescript', 'astigmatic', 'tearRate']
	lensesTree = trees.createTree(lenses,lensesLabels)
	treeplotter.createPlot(lensesTree)