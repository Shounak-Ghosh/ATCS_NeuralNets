import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Scanner;

/**
 * MAIN class
 * 
 * Tests a connectivity Model defined by Model.java.
 * The main method requires no args to run and will help the user configure, run, and train a model.
 * 
 * The waitForUserInput method pauses execution until the user inputs a String into the console.
 * A substring of the String containing all characters before the first whitespace character, or
 * the whole String if there are no whitespace characters, is returned. Thus, user input must not
 * contain any whitespace characters.
 * 
 * The parseInput method takes in a String representing the desired activations which the user input
 * into the console when prompted. The String is parsed and an array of doubles containing each activation
 * value is returned.
 * 
 * The main method, which is also the main method of the project, prompts the user for inputs and assists
 * them in using the Model. The user can run the model, train the model, change which model is being used,
 * or stop the program through inputs. For each mode, the user is prompted for input parameters and output
 * messages and values are printed to the console. In short, everything related to working with the Model
 * is done through this main method.
 * 
 * @author Jatin Kohli
 * @since 9/11/19
 */
public class ModelTester 
{
   private static final String SEP = File.separator; //System-dependent file path seperator string

   private static final String DEFAULT_INPUT_FILE = "Model_Inputs.txt";
   private static final String DEFAULT_OUTPUT_FILE = "Model_Outputs.txt";
   private static final String DEFAULT_OUTPUT_BMP = "bmps" + SEP + "Model_Output.bmp";

   private static Scanner s;

   /**
    * Waits for a String to be put into the console and then reads it.
    * Will pause execution of the current thread until input has been recieved. 
    *
    * @return The String that was put into the console
    */
   private static String waitForUserInput()
   {
      while(!s.hasNext())
      {
         try 
         {
            Thread.sleep(1);
            //Sleep for 1 ms until input is given
         } 
         catch (Exception e) 
         {
            e.printStackTrace();
         }
      } //while(!s.hasNext())

      return s.next();
   } //private static String waitForStringInput()

   /**
    * Parses activation inputs from a string.
    *
    * @param inputStr
    *       The input representing the activations for a model, seperated by commas.
    *       The input String must have no spaces, and each activation value must be seperated
    *       by a comma and nothing else (no white space).
    *       
    * @return
    *       The double[] with all of the activations specified in inputStr. The array's length is determined
    *       by how many activations (or one plus the number of commas) present in the input String.
    */
   private static double[] parseInput(String inputStr)
   {
      String inputStrCopy = inputStr;
      int commaIndex = inputStrCopy.indexOf(",");

      List<Double> inputs = new ArrayList<Double>();

      while (commaIndex != -1) 
      {
         double input = new Double(inputStrCopy.substring(0, commaIndex));
         inputs.add(input);

         inputStrCopy = inputStrCopy.substring(commaIndex + 1); //Take previously read input out of string
         commaIndex = inputStrCopy.indexOf(","); //Update commaIndex from new string
      }

      inputs.add(new Double(inputStrCopy)); //Add the last activation not added from the above loop

      double[] inputArray = new double[inputs.size()];
      int index = 0;

      for (double input : inputs)
      {
         inputArray[index] = input;
         index++;
      }

      return inputArray;
   } //private static double[] parseInput(String inputStr)

   /**
    * MAIN method
    *
    * Accepts user inputs and controls training, execution, and configuration of a model.
    * 
    * @param args Args for the Main method
    */
   public static void main(String[] args) 
   {
      s = new Scanner(System.in);

      Model model = null;

      String input = "change"; //Variable for user input, Set to "change" by default for initial setup

      boolean shouldContinue = true; //True by default, becomes false when the user wants to stop

      while (shouldContinue)
      {
         if (input.equals("run"))
         {
            double[] inputs = null;

            System.out.println(
                  "\r\nWould you like to input each activation value manually, " + 
                  "input the path to the file with all of the activation values, \r\n"+ 
                  "or take in a BMP file as input (and output a BMP file as output)? \r\n" + 
                  "Type 'manual' for manual input, 'file' for file input, or 'bmp' for BMP input."
            );
            
            input = waitForUserInput();

            while (!input.equals("manual") && !input.equals("file") && !input.equals("bmp"))
            {
               System.out.println(
                     "\r\nType 'manual' for manual input, 'file' for file input, or 'bmp' for BMP input."
               );

               input = waitForUserInput();
            }

            boolean isBmp = input.equals("bmp");

            BMPUtil dd = null; //BMPUtil object declared with required scope, will be instantiated if used

            if (input.equals("manual"))
            {
               System.out.println(
                     "\r\nEnter the inputs for the model(s) to have, seperated by commas and " +
                     "nothing else. \r\n(i.e. '0.2,-1.2' as inputs for a model with two input activations)"
               );
            
               inputs = parseInput(waitForUserInput());
            }
            else //input is "file" or "bmp"
            {
               System.out.println("\r\nWhat is the relative path for the input file?");

               String inFileName = waitForUserInput();
               File inputFile = new File(inFileName);

               while (!inputFile.exists())
               {
                  System.out.println(
                        "\r\nFile not found at " + inputFile.getPath() + 
                        ", enter a valid file path (i.e BMPUtil_10by10A_Pels.txt)"
                  );

                  inFileName = waitForUserInput();
                  inputFile = new File(inFileName);
               }

               if (isBmp)
               {
                  System.out.println(
                        "\r\nWould you like to convert the BMP to grayscale before running? (Type 'y' or 'n')"
                  );
 
                  boolean grayScale = waitForUserInput().equals("y");

                  dd = new BMPUtil(inFileName);
                  dd.printPelsToFile(DEFAULT_INPUT_FILE, grayScale);

                  inputFile = new File(DEFAULT_INPUT_FILE); //Set pel file as input file
               } //if (isBmp)

               double[] modelInputs = new double[model.scanner.inputNodes];

               Scanner inputFileScanner = null;

               try 
               {
                  inputFileScanner = new Scanner(inputFile);
               } 
               catch (FileNotFoundException e) 
               {
                  System.err.println("\r\nInput file not found\r\n" + e.getStackTrace());
               }

               try
               {
                  for (int numInput = 0; numInput < modelInputs.length; numInput++)
                  {
                     modelInputs[numInput] = inputFileScanner.nextDouble();
                  }
               }
               catch (NoSuchElementException e)
               {
                  System.err.println("\r\nInput file did not contain enough activation values for the model\r\n" + 
                        e.getStackTrace()
                  );
               }

               inputs = modelInputs;
            } //else

            boolean saveToFile = isBmp; //automatically save to file if the user inputs a BMP

            if (!saveToFile) //If the user did not input a bmp, check if they want to save outputs to a file
            {
               System.out.println("\r\nWould you like to print the outputs to a file? (Type 'y' or 'n')\r\n" + 
                  "If inputs are not printed to a file, they will be printed to the console");
            
               input = waitForUserInput();
               saveToFile = input.equals("y");
            }

            model.run(inputs);

            double[] outputs = new double[model.numOutputNodes];

            for (int output = 0; output < model.numOutputNodes; output++)
               outputs[output] = model.activations[model.outputIndex][output];

            if (saveToFile)
            {
               File outputFile = new File(DEFAULT_OUTPUT_FILE);
               
               try 
               {
                  outputFile.createNewFile();
               } 
               catch (IOException e) 
               {
                  e.printStackTrace();
               }

               try 
               {
                  PrintStream printer = new PrintStream(outputFile);

                  for (double output : outputs)
                  {
                     printer.print(output + " ");
                  }
               } 
               catch (FileNotFoundException e)
               {
                  e.printStackTrace();
               }

               System.out.println("\r\nOutputs succesfully printed to 'Model_Outputs.txt'");
            } //if (saveToFile)
            else //print to console instead of saving to file
            {
               System.out.println("\r\nModel returned:");

               for (int output = 0; output < model.numOutputNodes; output++)
                  System.out.println("\t" + outputs[output]);
               
               System.out.println("\r\nFor inputs:");

               for (double usedInput : inputs)
                  System.out.println("\t" + usedInput);
            } //else
         } //if (input.equals("run"))
         else if (input.equals("train"))
         {
            System.out.println("\r\nRandomize the weights? (y/n)");
            boolean randomWeights = waitForUserInput().equals("y");

            System.out.println("\r\nSave the new weights? (y/n)");
            boolean saveWeights = waitForUserInput().equals("y");

            int printPeriod = 0; //Default to no print statements

            System.out.println(
                  "\r\nType in how often you would like to print the current number of iterations " +
                  "during training. \r\n" + 
                  "In other words, how many iterations would you like between " + 
                  "each print statement (Type zero or any negative integer to disable prints during training)."
            );

            boolean isPrintPeriodValid = false;

            while (!isPrintPeriodValid)
            {
               input = waitForUserInput();

               try
               {
                  printPeriod = new Integer(input);

                  isPrintPeriodValid = true; //Only set to true after input can be parsed as an integger
               }
               catch (NumberFormatException e) //User input was not a valid integer
               {
                  System.out.println(
                        "\r\nType a valid integer for how often you would like print statements during training " + 
                        "(Type zero or any negative number to disable prints during training).");
               }
            } //while (!isPrintPeriodValid)

            System.out.println("\r\nWould you like to print the inputs/outputs for each test case? (y/n)");

            boolean printResults = waitForUserInput().equals("y");

            model.train(randomWeights, saveWeights, printPeriod, printResults);
         } //else if (input.equals("train"))
         else if (input.equals("change"))
         {
            System.out.println("\r\nWhat is the relative path for the config file?");

            File file = new File(waitForUserInput());

            while (!file.exists())
            {
               System.out.println("\r\nFile not found at " + file.getPath() + 
                     ", enter a valid file path (i.e models" + SEP + "TemplateConfig.txt)");

               file = new File(waitForUserInput());
            }

            System.out.println("\r\nWould you like to debug this model by printing out all activations? (y/n)");

            boolean debug = waitForUserInput().equals("y");

            model = new Model(file, debug);
         } //else if (input.equals("change"))
         else if (input.equals("stop"))
         {
            shouldContinue = false;
         }
         else
         {
            System.out.println("\r\nType one of the valid options:");
         }

         if (shouldContinue) //If the user did not input 'stop', print prompt for next loop
         {
            System.out.println("\r\nWould you like to run, train, change models, or stop? " +
               "(type 'run', 'train', 'change', or 'stop')");

            input = waitForUserInput();
         }
      } //while (shouldContinue)
   } //public static void main(String[] args) 
} //public class ModelTester 