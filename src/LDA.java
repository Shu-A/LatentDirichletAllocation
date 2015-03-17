import java.util.*;
import java.io.*;
import java.math.BigDecimal;

import org.apache.commons.cli.*;
import org.json.*;

class pComp implements Comparable<pComp> {
  int id;
  double p;
  @Override
  public int compareTo(pComp n) {
    return Double.compare(p, n.p);
  }
}

public class LDA {

  int D;     // The Number of Document
  int K;     // The Number of Topic
  int N;     // The Number of 
  int V;     // The Number of Vocablary

  List<Token> tokens;
  List<String> voca;
  int n_m_z[][];
  int n_z_t[][];
  int n_z[];
  int  z_n[];

  int seed;
  double alpha, beta;
  Random rand;

  public LDA(int docNum, int topicNum, double alpha, double beta,
             List<Token> tokens, List<String> voca, int seed) {
    this.D = docNum;
    this.K = topicNum;
    this.V = voca.size();
    this.N = tokens.size();
    this.n_m_z = new int[docNum][topicNum];
    this.n_z_t = new int[topicNum][this.V];
    this.n_z = new int[topicNum];
    this.z_n = new int[this.N];
    this.alpha = alpha;
    this.beta = beta;
    this.seed = seed;
    this.rand = new Random(seed);
    this.tokens = tokens;
    this.voca = voca;
    init();
  }

  private void init() {
    Random initRand = new Random(this.seed);
    int z;
    for (int i = 0; i < this.N; i++) {
      z = initRand.nextInt(K);
      this.z_n[i] = z;
      this.n_m_z[this.tokens.get(i).docId][z] += 1;
      this.n_z_t[z][this.tokens.get(i).wordId] += 1;
      this.n_z[z] += 1;
    }
  }

  private void sampler(int n) {

    double[] p_z = new double[this.K];

    int z = this.z_n[n];
    this.n_m_z[tokens.get(n).docId][z] -= 1;
    this.n_z_t[z][tokens.get(n).wordId] -= 1;
    this.n_z[z] -= 1;

    for (int k = 0; k < this.K; k++) {
      p_z[k] = (this.n_z_t[k][tokens.get(n).wordId] + beta)
                 * (this.n_m_z[tokens.get(n).docId][k] + alpha) / (this.n_z[k] + V * beta);
    }

    int new_z = sampling_multinomial(K, p_z);
    this.n_m_z[tokens.get(n).docId][new_z] += 1;
    this.n_z_t[new_z][tokens.get(n).wordId] += 1;
    this.n_z[new_z] += 1;
    this.z_n[n] = new_z;
  }

  private int sampling_multinomial(int n, double[] p) {
    double[] q = new double[n];
    q[0] = p[0];
    for (int k = 1; k < n; k++) {
      q[k] = q[k - 1] + p[k];
    }
    double u = this.rand.nextDouble() * q[n - 1];
    for (int k = 0; k < n; k++) {
      if (u < q[k]) {
        return k;
      }
    }
    return n - 1;
  } 

  public void inferrence() {
    for (int i = 0; i < this.N; i++) {
      sampler(i);
    }
  }

 public double[][] getPhi() {
    double phi[][] = new double[this.K][this.V];
    double Vbeta = this.beta + this.V;
    for (int i = 0; i < this.K; i++) {
      double sum = 0.0;
      int n_z = 0;
      for (int j = 0; j < this.V; j++) {
        n_z += n_z_t[i][j];
      }
      for (int j = 0; j < this.V; j++) {
        phi[i][j] = this.beta + this.n_z_t[i][j];
        phi[i][j] /= (n_z + Vbeta);
        sum += phi[i][j];
      }
      for (int j = 0; j < this.V; j++) {
        phi[i][j] /= sum;
      }
    }
    return phi;
  }

  public double[][] getTheta() {
    double theta[][] = new double[this.D][this.K];
    double Kalpha = this.alpha + this.K;
    for (int i = 0; i < this.D; i++) {
      double sum = 0.0;
      int n_m = 0;
      for (int j = 0; j < this.K; j++) {
        n_m += n_m_z[i][j];
      }
      for (int j = 0; j < this.K; j++) {
        theta[i][j] = this.alpha + this.n_m_z[i][j];
        theta[i][j] /= (n_m + Kalpha);
        sum += theta[i][j];
      }
      for (int j = 0; j < this.K; j++) {
        theta[i][j] /= sum;
      }
    }
    return theta;
  }

  public double perplexity() {
    double[][] phi = getPhi();
    double[][] theta = getTheta();
    double log_per = 0;
    for (int i = 0; i < this.N; i++) {
      double inner = 0.0;
      for (int j = 0; j < this.K; j++) {
        inner += phi[j][this.tokens.get(i).wordId]
                                    * theta[this.tokens.get(i).docId][j];
      }
      log_per -= Math.log(inner);
    }
    return Math.exp(log_per / this.N);
  }

  public int mergeTopics(double thre) {
    int[][] n_m_z = new int[this.D][this.K - 1];
    int[][] n_z_t = new int[this.K - 1][this.V];
    int[] n_z = new int[this.K - 1];
    double min = thre;
    int n1 = -1;
    int n2 = -1;
    double[][] phi = getPhi();
    for (int i = 0; i < this.K; i++) {
      for (int j = i + 1; j < this.K; j++) {
        double div = JSdivTopics(phi[i], phi[j], this.V);
        if (div < min ) {
          n1 = i;
          n2 = j;
          min = div;
        }
      }
    }
    if (n1 == -1) return 1;
    int k = 0;
    for (int i = 0; i < this.K; i++) {
      if (i == n2) {
        k = n1;
      } else if (i > n2) {
        k = i - 1;
      } else {
        k = i;
      }
      for (int j = 0; j < this.D; j++) {
        n_m_z[j][k] += this.n_m_z[j][i];
      }
      for (int j = 0; j < this.V; j++) {
        n_z_t[k][j] += this.n_z_t[i][j];
      }
      n_z[k] += this.n_z[i];
    }
    for (int i = 0; i < this.N; i++) {
      if (z_n[i] == n2) {
        this.z_n[i] = n1;
      } else if (z_n[i] > n2) {
        this.z_n[i]--;
      }
    }
    this.K--;
    this.n_m_z = n_m_z;
    this.n_z_t = n_z_t;
    this.n_z = n_z;

    return 0;

  }

  private double JSdivTopics(double[] p, double[] q, int n) {
    double[] m = averageTopics(p, q, n);
    return 0.5 * (KLdivTopics(p, m, n) + KLdivTopics(q, m, n));
  }

  private double KLdivTopics(double[] p, double[] q, int n) {
    double div = 0.0;
    for(int i = 0; i < n; i++) {
      div += p[i] * Math.log(p[i]/q[i]);
    }
    return div;
  }

  private double[] averageTopics(double[] p, double[] q, int n) {
    double[] avg = new double[n];
    for (int i = 0; i < n; i++) {
      avg[i] = (p[i] + q[i]) * 0.5;
    }
    return avg;
  }

  public static void main(String[] args) throws Exception {

    final int DEFAULT_NUM_TOPIC = 50;
    final double DEFAULT_ALPHA = 50.0 / DEFAULT_NUM_TOPIC;
    final double DEFAULT_BETA = 0.1;
    final int DEFAULT_SEED = 777;

    Options options = new Options();
    Option wordDocCnt = OptionBuilder.hasArg(true)
                                    .withArgName("file")
                                    .isRequired(true)
                                    .withDescription("File path of word-doc count list\n" + 
                                                     "  ## file format ##\n" +
                                                     "  <Number of documents>\n" +
                                                     "  <Number of vocablaries>\n" +
                                                     "  <Number of following lies>\n" +
                                                     "    (= line count of this file - 3)>\n" + 
                                                     "  <Document id> <Word id> <Word count>\n" + 
                                                     "  ... ")
                                    .create("w");
    options.addOption(wordDocCnt);

    Option vocablary = OptionBuilder.hasArg(true)
                                    .withArgName("file")
                                    .isRequired(true)
                                    .withDescription("File path of vocablary list\n" +
                                                     "  This file is list of unique words.")
                                    .create("v");
    options.addOption(vocablary);

    Option document = OptionBuilder.hasArg(true)
                                    .withArgName("file")
                                    .isRequired(true)
                                    .withDescription("File path of document list")
                                    .create("d");
    options.addOption(document);

    Option output = OptionBuilder.hasArg(true)
                                    .withArgName("file")
                                    .isRequired(true)
                                    .withDescription("File path of output")
                                    .create("o");
    options.addOption(output);

    Option iteration = OptionBuilder.hasArg(true)
                                    .withArgName("iteration")
                                    .isRequired(true)
                                    .withDescription("Number of iteration of Gibbs Sampling")
                                    .create("i");
    options.addOption(iteration);

    Option nTopic = OptionBuilder.hasArg(true)
                                    .withArgName("topics")
                                    .isRequired(false)
                                    .withDescription("Number of topics (default: 50)")
                                    .create("t");
    options.addOption(nTopic);

    Option alpha = OptionBuilder.hasArg(true)
                                    .withArgName("alpha")
                                    .isRequired(false)
                                    .withDescription("Hyperparameter 'alpha' in LDA model (default: 50/Num of Topic)")
                                    .create("a");
    options.addOption(alpha);

    Option beta = OptionBuilder.hasArg(true)
                                    .withArgName("beta")
                                    .isRequired(false)
                                    .withDescription("Hyperparameter 'beta' in LDA model (default: 0.1)")
                                    .create("b");
    options.addOption(beta);

    Option seed = OptionBuilder.hasArg(true)
                                    .withArgName("seed")
                                    .isRequired(false)
                                    .withDescription("Random seed (default: 777)")
                                    .create("s");
    options.addOption(seed);

    Option mrgThreshold = OptionBuilder.hasArg(true)
                                    .withArgName("threshold")
                                    .isRequired(false)
                                    .withDescription("Threshold of distance between for Topic merge\n(default: false)")
                                    .create("m");
    options.addOption(mrgThreshold);

    CommandLineParser parser = new PosixParser();
    CommandLine cmd = null;
    try {
      cmd = parser.parse(options, args);
    } catch (ParseException e) {
      HelpFormatter help = new HelpFormatter();
      help.printHelp("LDA", options, true);
      return;
    }

    String wordDocCntFile = cmd.getOptionValue("w");
    String docListFile = cmd.getOptionValue("d");
    String vocablaryFile = cmd.getOptionValue("v");
    String outputFile = cmd.getOptionValue("o");
    int ite = Integer.parseInt(cmd.getOptionValue("i"));

    int nt = DEFAULT_NUM_TOPIC;
    if (cmd.hasOption("t")) {
      nt = Integer.parseInt(cmd.getOptionValue("t"));
    }
    double pa = DEFAULT_ALPHA;
    if (cmd.hasOption("a")) {
      pa = Double.parseDouble(cmd.getOptionValue("a"));
    }
    double pb = DEFAULT_BETA;
    if (cmd.hasOption("b")) {
      pb = Double.parseDouble(cmd.getOptionValue("b"));
    }
    int sd = DEFAULT_SEED;
    if (cmd.hasOption("s")) {
      sd = Integer.parseInt(cmd.getOptionValue("s"));
    }
    double mt = 0.0;
    if (cmd.hasOption("m")) {
      mt = Double.parseDouble(cmd.getOptionValue("m"));
    }

    Scanner sc = new Scanner(new File(wordDocCntFile));
    int D = sc.nextInt();
    int V = sc.nextInt();
    int L = sc.nextInt();

    List<Token> tokens = new ArrayList<Token>();
    for (int i = 0; i < L; i++) {
      int dId = sc.nextInt() - 1;
      int wId = sc.nextInt() - 1;
      int cnt = sc.nextInt();
      for (int j = 0; j < cnt; j++) {
        Token t = new Token(dId, wId);
        tokens.add(t);
      }
    }

    List<String> voca = new ArrayList<String>();
    sc = new Scanner(new File(vocablaryFile));
    while (sc.hasNext()) {
      voca.add(sc.nextLine());
    }

    List<String> docs = new ArrayList<String>();
    sc = new Scanner(new File(docListFile));
    while (sc.hasNext()) {
      docs.add(sc.nextLine());
    }

    LDA lda = new LDA(D, nt, pa, pb, tokens, voca, sd);
    System.out.println("The number of Documents = " + lda.D);
    System.out.println("The number of Vocablary = " + lda.V);
    System.out.println("The number of Words     = " + lda.N);
    System.out.println("The number of Topics    = " + lda.K);
    System.out.println("Hyper parameter: alpha  = " + lda.alpha);
    System.out.println("Hyper parameter: beta   = " + lda.beta);
    System.out.println("The number of Iterations= " + ite);
    for (int i = 0; i < ite; i++) {
      lda.inferrence();
      System.out.println("iteration = " + (i + 1) + ": \tperplexity = " + lda.perplexity());
      if (mt != 0.0) {
        lda.mergeTopics(mt);
      }
    }
    try {
      File f = new File(outputFile);
      FileWriter fw = new FileWriter(f);
      BufferedWriter bw = new BufferedWriter(fw);
      PrintWriter pw = new PrintWriter(bw);

      JSONObject json = new JSONObject();

      JSONObject[] wordsJson = new JSONObject[lda.K];
      descWordDist(lda, wordsJson);
      json.put("word_distribution", wordsJson);

      JSONObject[] topicsJson = new JSONObject[lda.D];
      descTopicDist(lda, docs, topicsJson);
      json.put("topic_distribution", topicsJson);

      json.put("num_topic",lda.K);

      int indentSpaces = 2;
      pw.println(json.toString(indentSpaces));
      pw.close();

    } catch(IOException e) {
      System.err.println(e);
    }
    System.out.println("\n---------------------------------------------------------------------\n");
  }

  private static void descWordDist(LDA lda, JSONObject[] wordsJson) {
    double[][] phi = lda.getPhi();
    int maxWords = 5;
    int K = phi.length;

    for (int i = 0; i < K; i++) {
      wordsJson[i] = new JSONObject();
      wordsJson[i].put("topic", i);
      int V = phi[i].length;
      pComp phi_z[] = new pComp[V];
      for (int j = 0; j < V; j++) {
        pComp pc = new pComp();
        pc.id = j;
        pc.p = phi[i][j];
        phi_z[j] = pc;
      }
      Arrays.sort(phi_z);
      String[] words = new String[maxWords];
      for (int j = 0; j < maxWords; j++) {
        BigDecimal bd = new BigDecimal(phi_z[V - 1 - j].p);
        BigDecimal hu = bd.setScale(4, BigDecimal.ROUND_HALF_UP);
        words[j] = lda.voca.get(phi_z[V - 1 - j].id);
      }
      wordsJson[i].put("words", words);
    }
  }

  private static void descTopicDist(LDA lda, List<String> docs, JSONObject[] topicsJson) {
    double[][] phi = lda.getPhi();
    double[][] theta = lda.getTheta();
    int D = theta.length;

    for (int i = 0; i < D; i++) {
      topicsJson[i] = new JSONObject();
      topicsJson[i].put("title", docs.get(i));
      int K = theta[i].length;
      pComp theta_m[] = new pComp[K];
      for (int j = 0; j < K; j++) {
        pComp pc = new pComp();
        pc.id = j;
        pc.p = theta[i][j];
        theta_m[j] = pc;
      }

      BigDecimal[] dists = new BigDecimal[K];
      for (int j = 0; j < lda.K; j++) {
        BigDecimal bd = new BigDecimal(theta_m[j].p);
        BigDecimal hu = bd.setScale(4, BigDecimal.ROUND_HALF_UP);
        dists[j] = hu;
      }
      topicsJson[i].put("distribution", dists);
    }
  }
}
