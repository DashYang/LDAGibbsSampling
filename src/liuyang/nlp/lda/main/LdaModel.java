package liuyang.nlp.lda.main;

/**Class for Lda model
 * @author yangliu
 * @blog http://blog.csdn.net/yangliuy
 * @mail yangliuyx@gmail.com
 */
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import liuyang.nlp.lda.com.FileUtil;
import liuyang.nlp.lda.conf.PathConfig;

public class LdaModel {

	int[][] doc;// word index array
	int V, K, M, GROUP;// vocabulary size, topic number, document number , group
						// number
	int[][] z; // topic label array
	int[][] g; // group label array

	float alpha; // doc-topic dirichlet prior parameter
	float beta; // topic-word dirichlet prior parameter

	int[][] nmk;// given document m, count times of topic k. M*K
	int[][] nkt;// given topic k, count times of term t. K*V
	int[][] ncm;// given group c, count times of document m.
	int[] nmkSum;// Sum for each row in nmk
	int[] nktSum;// Sum for each row in nkt
	int[] ncmSum;// Sum for each row in ncm
	double[][] phi;// Parameters for topic-word distribution K*V
	double[][] theta;// Parameters for doc-topic distribution M*K
	double [][] gamma;// Parameters for group-doc distribution GROUO*M
	int iterations;// Times of iterations
	int saveStep;// The number of iterations between two saving
	int beginSaveIters;// Begin save model at this iteration

	double perplexity; // my parameter
	double lamda; // my parameter

	public LdaModel(LdaGibbsSampling.modelparameters modelparam) {
		// TODO Auto-generated constructor stub
		alpha = modelparam.alpha;
		beta = modelparam.beta;
		iterations = modelparam.iteration;
		K = modelparam.topicNum;
		saveStep = modelparam.saveStep;
		beginSaveIters = modelparam.beginSaveIters;
		GROUP = modelparam.group;
		lamda = modelparam.lamda;
	}

	public void initializeModel(Documents docSet) {
		// TODO Auto-generated method stub
		M = docSet.docs.size();
		V = docSet.termToIndexMap.size();
		nmk = new int[M][K];
		nkt = new int[K][V];

		// ncm
		ncm = new int[GROUP][M];
		// ncm

		nmkSum = new int[M];
		nktSum = new int[K];
		// ncmSum 
		ncmSum = new int[GROUP];
		// ncmSum
		phi = new double[K][V];
		theta = new double[M][K];
		gamma = new double [GROUP][M];
		// initialize documents index array
		doc = new int[M][];
		for (int m = 0; m < M; m++) {
			// Notice the limit of memory
			int N = docSet.docs.get(m).docWords.length;
			doc[m] = new int[N];
			for (int n = 0; n < N; n++) {
				doc[m][n] = docSet.docs.get(m).docWords[n];
			}
		}

		// initialize topic lable z for each word
		z = new int[M][];
		g = new int[M][];

		for (int m = 0; m < M; m++) {
			int N = docSet.docs.get(m).docWords.length;
			z[m] = new int[N];
			g[m] = new int[N];
			for (int n = 0; n < N; n++) {
				int initTopic = (int) (Math.random() * K);// From 0 to K - 1
				int initGroup = (int) (Math.random() * GROUP);// From 0 to K - 1
				z[m][n] = initTopic;
				g[m][n] = initGroup;

				// number of words in doc m assigned to topic initTopic add 1
				nmk[m][initTopic]++;
				// number of terms doc[m][n] assigned to topic initTopic add 1
				nkt[initTopic][doc[m][n]]++;
				// total number of words assigned to topic initTopic add 1
				nktSum[initTopic]++;

				// my group init
				ncm[initGroup][m]++;
				ncmSum[initGroup]++;
				// my group init
			}
			// total number of words in document m is N
			nmkSum[m] = N;
		}

	}

	public void inferenceModel(Documents docSet) throws IOException {
		// TODO Auto-generated method stub
		if (iterations < saveStep + beginSaveIters) {
			System.err
					.println("Error: the number of iterations should be larger than "
							+ (saveStep + beginSaveIters));
			System.exit(0);
		}
		for (int i = 0; i < iterations; i++) {
			System.out.println("Iteration " + i);
			if ((i >= beginSaveIters)
					&& (((i - beginSaveIters) % saveStep) == 0)) {
				// Saving the model
				System.out.println("Saving model at iteration " + i + " ... ");
				// Firstly update parameters
				updateEstimatedParameters();
				// Secondly print model variables
				saveIteratedModel(i, docSet);
			}

			// Use Gibbs Sampling to update z[][]
			for (int m = 0; m < M; m++) {
				int N = docSet.docs.get(m).docWords.length;
				for (int n = 0; n < N; n++) {
					// Sample from p(z_i|z_-i, w)
					int newTopic = sampleTopicZ(m, n);
					z[m][n] = newTopic;
					int newGroup = sampleGroupZ(m, n);
					g[m][n] = newGroup;
				}
			}
		}
	}

	private void updateEstimatedParameters() {
		// TODO Auto-generated method stub
		for (int k = 0; k < K; k++) {
			for (int t = 0; t < V; t++) {
				phi[k][t] = (nkt[k][t] + beta) / (nktSum[k] + V * beta);
			}
		}

		for (int m = 0; m < M; m++) {
			for (int k = 0; k < K; k++) {
				theta[m][k] = (nmk[m][k] + alpha) / (nmkSum[m] + K * alpha);
			}
		}
		
		for (int c = 0 ; c < GROUP ; c ++) {
			for(int m = 0 ; m < M; m ++) {
				gamma[c][m] = 0.0;
				for (int k = 0; k < K; k++) {
					for (int t = 0; t < V; t++) {
						gamma[c][m] += phi[k][t] * theta[m][k] 
								* (ncm[c][m] + lamda) / (ncmSum[c] + M * lamda) ;
					}
				}
			}
		}
	//	 get perplexity begin
		 double perplexity_fenzi = 0.0;
		 double perplexity_fenmu = 0.0;
		 System.out.println("topic " + K );
		 for(int m = 0 ; m < M ; m++ ) {
			 double multiresult = 1.0;
			 for(int t = 0 ; t < V ; t ++) {
				 double addresult = 0.0;
				 for(int k = 0 ; k < K; k++) {
					 addresult += phi[k][t] * theta[m][k];
				 }
				 perplexity_fenzi += - Math.log(addresult) / Math.log(2.0);
				 System.out.println(perplexity_fenzi);
				 if(multiresult <= 0.0)
					 System.exit(0);
			 }		
			 perplexity_fenmu += nmkSum[m];
		 }
		 perplexity =  Math.pow(Math.E, perplexity_fenzi / perplexity_fenmu);
		 //get perlexity end
	}

	private int sampleTopicZ(int m, int n) {
		// TODO Auto-generated method stub
		// Sample from p(z_i|z_-i, w) using Gibbs upde rule

		// Remove topic label for w_{m,n}
		int oldTopic = z[m][n];
		nmk[m][oldTopic]--;
		nkt[oldTopic][doc[m][n]]--;
		nmkSum[m]--;
		nktSum[oldTopic]--;

		// Compute p(z_i = k|z_-i, w)
		double[] p = new double[K];
		for (int k = 0; k < K; k++) {
			p[k] = (nkt[k][doc[m][n]] + beta) / (nktSum[k] + V * beta)
					* (nmk[m][k] + alpha) / (nmkSum[m] + K * alpha);
		}

		// Sample a new topic label for w_{m, n} like roulette
		// Compute cumulated probability for p
		for (int k = 1; k < K; k++) {
			p[k] += p[k - 1];
		}
		double u = Math.random() * p[K - 1]; // p[] is unnormalised
		int newTopic;
		for (newTopic = 0; newTopic < K; newTopic++) {
			if (u < p[newTopic]) {
				break;
			}
		}

		// Add new topic label for w_{m, n}
		nmk[m][newTopic]++;
		nkt[newTopic][doc[m][n]]++;
		nmkSum[m]++;
		nktSum[newTopic]++;
		return newTopic;
	}

	private int sampleGroupZ(int m, int n) {
		// TODO Auto-generated method stub
		// Sample from p(z_i|z_-i, w) using Gibbs upde rule

		// Remove group label for f_{m,n}
		int oldGroup = g[m][n];
		ncm[oldGroup][m]--;
		ncmSum[oldGroup]--;

		// Remove topic label for w_{m,n}
		int oldTopic = z[m][n];
		nmk[m][oldTopic]--;
		nkt[oldTopic][doc[m][n]]--;
		nmkSum[m]--;
		nktSum[oldTopic]--;

		// Compute p(z_i = k|z_-i, w)
		double p[] = new double[GROUP];
		for (int c = 0; c < GROUP; c++) {
			p[c] = 0.0;
			for (int k = 0; k < K; k++) {
//				System.out.println(ncm[c][m] + " " + ncmSum[c]);
				p[c] += (nkt[k][doc[m][n]] + beta) / (nktSum[k] + V * beta)
						* (nmk[m][k] + alpha) / (nmkSum[m] + K * alpha)
						* (ncm[c][m] + lamda) / (ncmSum[c] + M * lamda);
			}
		}
		
		// Sample a new topic label for w_{m, n} like roulette
		// Compute cumulated probability for p
		for (int c = 1; c < GROUP; c++) {
			p[c] += p[c - 1];
		}
		
		if(p[GROUP - 1] < 0.0) {
			System.out.println(p[GROUP - 1]);
			for(int c = 0; c < GROUP ; c++ ) {
					System.out.print(" " + ncm[c][m]);
			}
			System.out.println("ncm");
			for (int c = 0; c < GROUP; c++) {
				System.out.print(" " + ncmSum[c]);
			}
			System.out.println("ncmsum");
			for (int c = 0; c < GROUP; c++) {
				System.out.print(" " + ncmSum[c]);
			}
			System.out.println("ncmsum");
			
		}
		
		double u = Math.random() * p[GROUP - 1]; // p[] is unnormalised
		int newGroup;
		
//		System.out.println("u:" + u);
		for (newGroup = 0; newGroup < GROUP; newGroup++) {
			if (u < p[newGroup]) {
				break;
			}
		}

		// Add new topic label for w_{m, n}
		nmk[m][oldTopic]++;
		nkt[oldTopic][doc[m][n]]++;
		nmkSum[m]++;
		nktSum[oldTopic]++;

		// add new group
		ncm[newGroup][m]++;
		ncmSum[newGroup]++;
		return newGroup;
	}

	public void saveIteratedModel(int iters, Documents docSet)
			throws IOException {
		// TODO Auto-generated method stub
		// lda.params lda.phi lda.theta lda.tassign lda.twords
		// lda.params
		String resPath = PathConfig.LdaResultsPath;
		String modelName = "lda_" + iters;
		ArrayList<String> lines = new ArrayList<String>();
		lines.add("alpha = " + alpha);
		lines.add("beta = " + beta);
		lines.add("topicNum = " + K);
		lines.add("docNum = " + M);
		lines.add("termNum = " + V);
		lines.add("iterations = " + iterations);
		lines.add("saveStep = " + saveStep);
		lines.add("beginSaveIters = " + beginSaveIters);
		// new line :perplexity
		lines.add("perplexity = " + perplexity);
		FileUtil.writeLines(resPath + modelName + ".params", lines);

		// lda.phi K*V
		BufferedWriter writer = new BufferedWriter(new FileWriter(resPath
				+ modelName + ".phi"));
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < V; j++) {
				writer.write(phi[i][j] + "\t");
			}
			writer.write("\n");
		}
		writer.close();

		// lda.ncm GROUP*M
		writer = new BufferedWriter(new FileWriter(resPath
				+ modelName + ".gamma"));
		for (int i = 0; i < GROUP; i++) {
			for (int j = 0; j < M; j++) {
				writer.write(gamma[i][j] + "\t");
			}
			writer.write("\n");
		}
		writer.close();

		// lda.theta M*K
		writer = new BufferedWriter(new FileWriter(resPath + modelName
				+ ".theta"));
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < K; j++) {
				writer.write(theta[i][j] + "\t");
			}
			writer.write("\n");
		}
		writer.close();

		// lda.tassign
		writer = new BufferedWriter(new FileWriter(resPath + modelName
				+ ".tassign"));
		for (int m = 0; m < M; m++) {
			for (int n = 0; n < doc[m].length; n++) {
				writer.write(doc[m][n] + ":" + z[m][n] + "\t");
			}
			writer.write("\n");
		}
		writer.close();

		// lda.twords phi[][] K*V
		writer = new BufferedWriter(new FileWriter(resPath + modelName
				+ ".twords"));
		int topNum = 20; // Find the top 20 topic words in each topic
		for (int i = 0; i < K; i++) {
			List<Integer> tWordsIndexArray = new ArrayList<Integer>();
			for (int j = 0; j < V; j++) {
				tWordsIndexArray.add(new Integer(j));
			}
			Collections.sort(tWordsIndexArray, new LdaModel.TwordsComparable(
					phi[i]));
			writer.write("topic " + i + "\t:\t");
			for (int t = 0; t < topNum; t++) {
				writer.write(docSet.indexToTermMap.get(tWordsIndexArray.get(t))
						+ " " + phi[i][tWordsIndexArray.get(t)] + "\t");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public class TwordsComparable implements Comparator<Integer> {

		public double[] sortProb; // Store probability of each word in topic k

		public TwordsComparable(double[] sortProb) {
			this.sortProb = sortProb;
		}

		@Override
		public int compare(Integer o1, Integer o2) {
			// TODO Auto-generated method stub
			// Sort topic word index according to the probability of each word
			// in topic k
			if (sortProb[o1] > sortProb[o2])
				return -1;
			else if (sortProb[o1] < sortProb[o2])
				return 1;
			else
				return 0;
		}
	}
}
