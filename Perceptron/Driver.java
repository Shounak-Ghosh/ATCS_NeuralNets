/**
 * Driver class used to run the Perceptron model
 * @author Shounak Ghosh
 * @version 4.16.2020
 */
public class Driver
{
   /*
    * Use 'javac' to compile
    * and 'java' to run
    *
    * ex: javac Perceptron.java Driver.java
    *     java  Driver (command-line args)
    */

   /**
    * Main method
    * @param args stores command line arguments.
    *             These arguments consist of custom input configuration filenames and output final weights filenames
    */
   public static void main(String[] args)
   {
      String infile = "TEST.txt"; // default input configuration filename
      String outfile = "OUT.txt"; // default output filename

      if (args.length > 0)
      {
         infile = args[0]; // infile taken in from command line
         if (args.length > 1)
         {
            outfile = args[1];
         }
      }

      Perceptron p = new Perceptron(infile); // Perceptron created
      p.trainPerceptron();

      try
      {
         p.writeWeights(outfile);
      }
      catch (Exception e)
      {
         e.printStackTrace();
      }

   } // public static void main(String[] args)
} // public class Driver
