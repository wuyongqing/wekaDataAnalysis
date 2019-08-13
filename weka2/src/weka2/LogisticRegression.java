package weka2;

import java.io.File;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import weka.classifiers.functions.Logistic;

import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import weka.core.converters.ArffLoader;

import weka.filters.Filter;

import weka.filters.unsupervised.attribute.Remove;

 

 

public class LogisticRegression {


    public static void main(String[] args) throws Exception {

       // TODOAuto-generated method stub

      

    	ArffLoader atf = new ArffLoader(); //Reads a source that is in arff (attribute relation file format) format.

 

        File inputFile = new File("wine.arff");//读入训练文件

        atf.setFile(inputFile);

        Instances instancesTrain = atf.getDataSet(); // 得到格式化的训练数据

       
        
        /*可用于过滤属性
        String[] removeOptions = new String[2]; 

        removeOptions[0] = "-R";                            // "range" 

        removeOptions[1] = "7";                             // 7th attribute去掉第7 个 

        Remove remove1 = new Remove();                // new instance of filter

        remove1.setOptions(removeOptions);                  // set options 

        remove1.setInputFormat(instancesTrain);  
		*/
        
        instancesTrain.setClassIndex(0);//设置类别位置
		
        inputFile = new File("wine_test.arff");//读入测试文件

        atf.setFile(inputFile);

        Instances instancesTest = atf.getDataSet(); // 得到格式化的测试数据

        //remove1.setInputFormat(instancesTest);

        //Instances newInstancesTest1=Filter.useFilter(instancesTest, remove1);//可得到新的测试数据

        instancesTest.setClassIndex(0); //设置分类属性所在行号（第一行为0号），instancesTest.numAttributes()可以取得属性总数

        //System.out.println("平均绝对误差："+eval.meanAbsoluteError());//越小越好

        //System.out.println("均方根误差："+eval.rootMeanSquaredError());//越小越好
 
       // System.out.println("相关性系数:"+eval.correlationCoefficient());//越接近1越好

        //System.out.println("根均方误差："+eval.rootMeanSquaredError());//越小越好

        //System.err.println("是否准确的参考值："+eval.meanAbsoluteError());//越小越好
        
        crossValidation(instancesTrain);
        System.out.println("-------------------------------------------------------------------------\n");
        myClassifier(instancesTrain);
        //evalueateTestData(instancesTest);
        
    }

    public static void crossValidation(Instances m_instances) throws Exception
    {
        J48 classifier=new J48();

        Evaluation eval=new Evaluation(m_instances);
        eval.crossValidateModel(classifier, m_instances, 10, new Random(1));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
    }
    
    /*测试
    public static void evalueateTestData(Instances m_instances) throws Exception
    {
        J48 classifier=new J48();
        classifier.buildClassifier(m_instances);
        
        Instance sample = m_instances.instance(0);
        Evaluation eval=new Evaluation(m_instances);
        eval.evaluateModel(classifier, m_instances);
        //System.out.println(eval.toClassDetailsString());
        //System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
    }
     */
    
    public static void myClassifier(Instances m_instances) throws Exception
    {
    	Logistic m_classifier=new Logistic();//Logistic用以建立一个逻辑回归分类器

        String options[]=new String[4];//训练参数数组

        options[0]="-R";//cost函数中的预设参数  影响cost函数中参数的模长的比重

        options[1]="1E-5";//设为1E-5

        options[2]="-M";//最大迭代次数

        options[3]="10";//最多迭代计算10次

        m_classifier.setOptions(options);
        m_classifier.buildClassifier(m_instances); //训练         
        
        Evaluation eval=new Evaluation(m_instances);
        eval.crossValidateModel(m_classifier, m_instances, 10, new Random(1));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
        System.out.println("-------------------------------------------------------------------------\n");
        System.out.println(m_classifier.toString());
    }

}