import java.io.*;
import java.util.Scanner;

/**
 * Takes in a BMP file and parses its header info and individual pel values. Large portions of this class are
 * taken from the DibDump class provided for ATCS Neural Networks by Dr. Nelson.
 * 
 * When a BMPUtil object is created with a BMP file, it will automatically call the readBitmapFromFile
 * method described below.
 * 
 * BMPUtil contains two private methods, swapInt and swapShort, to change the order of the bits in the
 * respective values to convert from little endian to big endian format. These are used both when reading
 * and creating bitmaps.
 * 
 * The private method called pelToRGB which converts an integer pel value into an RBGQuad
 * object containing its red, green, and blue components as integers. The resersved byte of the pel is
 * discarded. BMPUtil contains a method to do the inverse called rgbToPel which takes in the red, green,
 * and blue components to construct an integer pel from.
 * 
 * The private method colorToGrayscale takes in an integer color pel and returns 
 * the equivalent integer grayscale pel.
 * 
 * The readBitMapFromFile method takes in a String file path to a BMP file and populates the BMPUtil's fields 
 * with data from the BMP's header and pel values. This method is called in the constructor with its String argument,
 * and changing the BMP file which a BMPUtil object corresponds to requires readBitmapFromFile to be called with the
 * String file path of the new BMP.
 * 
 * The printPelsToFile method can be used to print just the BMP's pel values to a file. Each row of pels will
 * be on its own line in the file, and each pel will be seperated with a space. Each pel value will be scaled
 * by a factor of 2^n, where n is the number of bits per pixel, before being printed to the file.
 * 
 * The readPelsFromFile method can be used to change the pel values but not the BMP header info of the 
 * BMPUtil object. This method must be used with a pel file generated from the printPelsToFile method. 
 * Like how the printPelsToFile method scales the pels down before printing, the readPelsFromFile method
 * scales each pel from the file up by the same factor. This method can be used to change the pixel values
 * of a BMP but preserve its header data, such as image size or bits per pixel.
 * 
 * The printBitmapToFile method will create a BMP with the same header data as the BMPUtil's previously read 
 * BMP and pel values from the BMPUtil's previously read pel file (or previously read BMP if no pel files 
 * have been read since a BMP was read). The BMP will be created at the file path with the name of the 
 * method's string parameter.
 * 
 * Lastly, the BMPUtil class has a main method which can create a pel file from a BMP. The main method requires
 * two arguments, where the first is the String file path to the input BMP and the second is the String file path
 * for the output pel file to be created at.
 * 
 * Excerpt from Dr. Nelson's DibDump about reading bitmaps:
 *
 *         The BMP format assumes an Intel integer type (little endian),
 *         however, the Java virtual machine uses the Motorola integer type (big
 *         endian), so we have to do a bunch of byte swaps to get things to read
 *         and write correctly. Also note that many of the values in a bitmap
 *         header are unsigned integers of some kind and Java does not know
 *         about unsigned values, except for reading in unsigned byte and
 *         unsigned short, but the unsigned int still poses a problem. We don't
 *         do any math with the unsigned int values, so we won't see a problem.
 *
 *         Bitmaps on disk have the following basic structure BITMAPFILEHEADER
 *         (may be missing if file is not saved properly by the creating
 *         application) BITMAPINFO - BITMAPINFOHEADER RGBQUAD - Color Table
 *         Array (not present for true color images) Bitmap Bits in one of many
 *         coded formats
 *
 *         The BMP image is stored from bottom to top, meaning that the first
 *         scan line in the file is the last scan line in the image.
 *
 *         For ALL images types, each scan line is padded to an even 4-byte
 *         boundary.
 * 
 *         For images where there are multiple pels per byte, the left side is
 *         the high order element and the right is the low order element.
 *
 *         in Windows on a 32 bit processor... DWORD is an unsigned 4 byte
 *         integer WORD is an unsigned 2 byte integer LONG is a 4 byte signed
 *         integer
 *
 *         in Java we have the following sizes:
 *
 *         byte 1 signed byte (two's complement). Covers values from -128 to
 *         127.
 *
 *         short 2 bytes, signed (two's complement), -32,768 to 32,767
 *
 *         int 4 bytes, signed (two's complement). -2,147,483,648 to
 *         2,147,483,647. Like all numeric types ints may be cast into other
 *         numeric types (byte, short, long, float, double). When lossy casts
 *         are done (e.g. int to byte) the conversion is done modulo the length
 *         of the smaller type.
 * 
 * @author Shounak Ghosh
 * @since June 1 2020
 */
public class BMPUtil
{
   private static final int LARGEST_COLOR_TABLE_SIZE = 256; //For creating colorPallet Array

   private static final int DEFAULT_BITS_PER_PEL = 24;
   private static final int DEFAULT_RESERVED_VALUE = 0;
   private static final int DEFAULT_OFFSET = 54; //byte offset for 24 bit images, used in file header
   private static final int DEFAULT_DEAD_BYTE = 0;

   private static final int TRUE_COLOR = 0;
   private static final int PEL_SIZE_WITH_RESERVED = 32; //Size of pel required to utilize reserved byte
   private static final int SCALE_FACTOR_32_BIT = 24; //Ignore reserved byte when scaling
   private static final int BI_RGB = 0;

   /**
    * Factor to scale pels when printing/reading from pel files such that the actual pel values
    * range from 0 to whatever the bits per pel of the bitmap is while the values in the pel file
    * always range from 0.0 to 1.0. The scale factor gets updated whenever a BMP is read using 
    * readBitmapFromFile and the bmpInfoHeader_biBitCount field is updated.
    */
   private double scaleFactor;

   //File Header values values
   private int bmpFileHeader_bfType;      //WORD
   private int bmpFileHeader_bfSize;      //DWORD
   private int bmpFileHeader_bfReserved1; //WORD
   private int bmpFileHeader_bfReserved2; //WORD
   private int bmpFileHeader_bfOffBits;   //DWORD

   //Info Header values
   private int bmpInfoHeader_biSize;          //DWORD
   private int bmpInfoHeader_biWidth;         //LONG
   private int bmpInfoHeader_biHeight;        //LONG
   private int bmpInfoHeader_biPlanes;        //WORD
   private int bmpInfoHeader_biBitCount;      //WORD
   private int bmpInfoHeader_biCompression;   //DWORD
   private int bmpInfoHeader_biSizeImage;     //DWORD
   private int bmpInfoHeader_biXPelsPerMeter; //LONG
   private int bmpInfoHeader_biYPelsPerMeter; //LONG
   private int bmpInfoHeader_biClrUsed;       //DWORD
   private int bmpInfoHeader_biClrImportant;  //DWORD

   public int[][] imageArray; //The true color pels

   /*
    * If bmpInfoHeader_biHeight is negative then the image is a top down DIB. This
    * flag is used to identify it as such. Note that when the image is saved, it will be written
    * out in the usual inverted format with a positive bmpInfoHeader_biHeight value.
    */
   private static boolean topDownDIB = false;

   /*
    * Temporary varibles used for instantiating the imageArray with the values of each pel, 
    * outputing the imageArray's contents to a bmp file, or both. These are declared once and
    * used multiple times, sometimes in multiple methods, for efficiency.
    */
   private int i, j, k;
   private int numberOfColors;
   private int pel;
   private int iByteVal, iColumn, iBytesPerRow, iPelsPerRow, iTrailingBits, iDeadBytes;
   private int rgbQuad_rgbBlue, rgbQuad_rgbGreen, rgbQuad_rgbRed, rgbQuad_rgbReserved;

   /**
    * Creates a new BMPUtil object
    *
    * @param inFileName 
    *       The String path for the BMP file for the BMPUtil to take in. Once the BMPUtil object is created, 
    *       it will automatically create a 2D pel array representing the RGB value of each pixel in the BMP. 
    *       This pel array can be accessed with the public static field called 'imageArray'
    */
   public BMPUtil(String inFileName)
   {
      readBitmapFromFile(inFileName);
   }

   /**
    * Converts an integer (DWORD, 4 bytes) from little endian to big endian format
    *
    * @param v 
    *       The integer (in little endian) to convert
    *
    * @return
    *       The integer v's equivalent in big endian format 
    */
   private static int swapInt(int v) 
   {
      return (v >>> 24) | (v << 24) | ((v << 8) & 0x00FF0000) | ((v >> 8) & 0x0000FF00);
   }

   /**
    * Converts a short (2 bytes) from little endian to big endian format
    *
    * @param v 
    *       The short (in little endian) to convert
    *
    * @return
    *       The short v's equivalent in big endian format 
    */
   private static int swapShort(int v) 
   {
      return ((v << 8) & 0xFF00) | ((v >> 8) & 0x00FF);
   }

   /**
    * Returns the red, green and blue colors of the picture element. The pel's reserved value is
    * ignored. See rgbToPel(int red, int green, int blue) to go the the other way.
    * 
    * @param pel
    *       The integer (32 bit) pel value containing the RGB values for the RgbQuad object to hold.
    *       The reserved value of the pel is ignored.
    *
    * @return 
    *       An RgbQuad object which contains the input pel's rgb values. The returned RgbQuad's reserved 
    *       value is always set to zero, regardless of the pel's reserved value.
    */
   private RgbQuad pelToRGB(int pel) 
   {
      RgbQuad rgb = new RgbQuad();

      rgb.reserved = 0;

      rgb.blue = pel & 0x00FF;
      rgb.green = (pel >> 8) & 0x00FF;
      rgb.red = (pel >> 16) & 0x00FF;

      return rgb;
   }

   /**
    * Takes red, green and blue color values plus and returns a single 32-bit integer color.
    * The reserved byte of the returned pel is not accounted for.
    *
    * @param red
    *       The red color value (byte) for the returned pel to have
    * 
    * @param green
    *       The green color value (byte) for the returned pel to have
    *
    * @param blue
    *       The blue color value (byte) for the returned pel to have
    *
    * @return 
    *       An integer (32 bit) pel with the desired red, green, and blue values.
    *       The reserve byte of the pel is not accounted for.
    */
   private int rgbToPel(int red, int green, int blue) 
   {
      return (red << 16) | (green << 8) | blue;
   }

   /**
    * Converts a color pel to a grayscale pel. 
    *
    * The conversion is done with one of many possible formulas:
    * Y = 0.3 * RED + 0.589 * GREEN + 0.11 * Blue 
    * 
    * Y represents the Red, Green, and Blue values for the returned grayscale pel
    *
    * @param pel
    *       The color pel to convert to Grayscale
    *
    * @return
    *       The grayscale version of the color pel
    */
   private int colorToGrayscale(int pel) 
   {
      RgbQuad rgb = pelToRGB(pel);

      double redVal = 0.3 * (double) rgb.red;
      double greenVal = 0.589 * (double) rgb.green;
      double blueVal = 0.11 * (double) rgb.blue;

      int lum = (int) Math.round(redVal + greenVal + blueVal);

      return rgbToPel(lum, lum, lum);
   }

   /**
    * Reads header and pel values from the given input BMP file.
    * These values are stored in this class's fields for printing pel files and bitmaps.
    *
    * @param inFileName
    *       The input BMP to read 
    */
   public void readBitmapFromFile(String inFileName)
   {
      int[] colorPallet = new int[LARGEST_COLOR_TABLE_SIZE]; //The color table

      try //lots of things can go wrong when doing file i/o
      {
         //Open the file that is the first command line parameter
         FileInputStream fstream = new FileInputStream(inFileName);

         //Convert our input stream to a DataInputStream
         DataInputStream in = new DataInputStream(fstream);

         /*
          * The following describes the purpose and format of each BMP file header value
          * 
          * bfType Specifies the file type. It must be set to the signature word BM
          * (0x4D42) to indicate bitmap. 
          *
          * bfSize Specifies the size, in bytes, of the bitmap file. 
          *
          * bfReserved1 and bfReserved2 are reserved Words 
          *
          * bfOffBits Specifies the offset, in bytes, from the BITMAPFILEHEADER
          * structure to the bitmap bits
          *
          * The code below reads the BMP file header section and converts the values to big endian
          */
         bmpFileHeader_bfType = swapShort(in.readUnsignedShort());      //WORD
         bmpFileHeader_bfSize = swapInt(in.readInt());                  //DWORD
         bmpFileHeader_bfReserved1 = swapShort(in.readUnsignedShort()); //WORD
         bmpFileHeader_bfReserved2 = swapShort(in.readUnsignedShort()); //WORD
         bmpFileHeader_bfOffBits = swapInt(in.readInt());               //DWORD

         /* 
          * The following describes the purpose and format of each BMP info header value
          *
          * biSize specifies the size of the structure, in bytes. This size does not
          * include the color table or the masks mentioned in the biClrUsed member. See
          * the Remarks section for more information. 
          *
          * biWidth specifies the width of the bitmap, in pixels. 
          *
          * biHeight specifies the height of the bitmap, in pixels. 
          * If biHeight is positive, the bitmap is a bottom-up DIB and its origin is the lower left corner. 
          * If biHeight is negative, the bitmap is a top-down DIB and its origin is the upper left corner. 
          * If biHeight is negative, indicating a top-down DIB, 
          * biCompression must be either BI_RGB or BI_BITFIELDS. Top-down DIBs cannot be compressed. 
          *
          * biPlanes specifies the number of planes for the
          * target device. This value must be set to 1. 
          * 
          * biBitCount Specifies the number of bits per pixel. The biBitCount member of the BITMAPINFOHEADER 
          * structure determines the number of bits that define each pixel and the maximum number
          * of colors in the bitmap. This member must be one of the following values:
          *
          *    1:  The bitmap is monochrome, and the bmiColors member
          *        contains two entries. Each bit in the bitmap array represents a pixel. The
          *        most significant bit is to the left in the image. If the bit is clear, the
          *        pixel is displayed with the color of the first entry in the bmiColors table.
          *        If the bit is set, the pixel has the color of the second entry in the table.
          *    
          *    2:  The bitmap has four possible color values. The most significant half-nibble
          *        is to the left in the image. 
          *
          *    4:  The bitmap has a maximum of 16 colors, and the bmiColors member contains up to 16 entries. 
          *        Each pixel in the bitmap is represented by a 4-bit index into the color table. 
          *        The most significant nibble is to the left in the image. For example, 
          *        if the first byte in the bitmap is 0x1F, the byte represents two pixels. 
          *        The first pixel contains the color in the second table entry, and the second pixel 
          *        contains the color in the sixteenth table entry. 
          * 
          *    8:  The bitmap has a maximum of 256 colors, 
          *        and the bmiColors member contains up to 256 entries. 
          *        In this case, each byte in the array represents a single pixel. 
          *
          *    16: The bitmap has a maximum of 2^16 colors. If the biCompression member of the 
          *        BITMAPINFOHEADER is BI_RGB, the bmiColors member is NULL. Each WORD in the bitmap array
          *        represents a single pixel. The relative intensities of red, green, and blue are 
          *        represented with 5 bits for each color component. 
          *        The value for blue is in the least significant 5 bits, 
          *        followed by 5 bits each for green and red. The most significant bit is not used. 
          *        The bmiColors color table is used for optimizing colors used on
          *        palette-based devices, and must contain the number of entries specified by
          *        the biClrUsed member of the BITMAPINFOHEADER. 
          * 
          *    24: The bitmap has a maximum of 2^24 colors, and the bmiColors member is NULL. 
          *        Each 3-byte triplet in the bitmap array represents the relative intensities of 
          *        blue, green, and red, respectively, for a pixel. The bmiColors color table is used 
          *        for optimizing colors used on palette-based devices, and must contain the number of 
          *        entries specified by the biClrUsed member of the BITMAPINFOHEADER.
          *    
          *    32: The bitmap has a maximum of 2^32 colors. If the biCompression member of the 
          *        BITMAPINFOHEADER is BI_RGB, the bmiColors member is NULL. Each DWORD in the bitmap array
          *        represents the relative intensities of blue, green, and red, respectively, for a pixel. 
          *        The high byte in each DWORD is not used. The bmiColors color table is used for optimizing 
          *        colors used on palette-based devices, and must contain the number of entries specified by 
          *        the biClrUsed member of the BITMAPINFOHEADER. 
          *        If the biCompression member of the BITMAPINFOHEADER is BI_BITFIELDS, the bmiColors member 
          *        contains three DWORD color masks that specify the red, green, and blue components, 
          *        respectively, of each pixel. Each DWORD in the bitmap array represents a single pixel. 
          *
          * biCompression specifies the type of compression for a compressed bottom-up bitmap (top-down
          * DIBs cannot be compressed). This member can be one of the following values:
          * 
          *    BI_RGB: An uncompressed format. BI_BITFIELDS Specifies that the bitmap is not compressed 
          *            and that the color table consists of three DWORD color masks that specify the 
          *            red, green, and blue components of each pixel. This is valid when used with 16- and 
          *            32-bpp bitmaps. This value is valid in Windows Embedded CE versions 2.0 and later. 
          *        
          *    BI_ALPHABITFIELDS: The bitmap is not compressed and the color table consists of four DWORD
          *                       color masks that specify the red, green, blue, and alpha components of each
          *                       pixel. This is valid when used with 16- and 32-bpp bitmaps. This value is
          *                       valid in Windows CE .NET 4.0 and later. You can OR any of the values in the
          *                       above table with BI_SRCPREROTATE to specify that the source DIB section has
          *                       the same rotation angle as the destination. 
          *
          * biSizeImage specifies the size, in bytes, of the image. This value will be the 
          * number of bytes in each scan line which must be padded to insure the line is a 
          * multiple of 4 bytes (it must align on a DWORD boundary) times the number of rows. 
          * This value may be set to zero for BI_RGB bitmaps (so you cannot be sure it will be set).
          * 
          * biXPelsPerMeter specifies the horizontal resolution, in pixels per meter, of
          * the target device for the bitmap. An application can use this value to select
          * a bitmap from a resource group that best matches the characteristics of the
          * current device. 
          *
          * biYPelsPerMeter specifies the vertical resolution, in pixels
          * per meter, of the target device for the bitmap.
          * 
          * biClrUsed specifies the number of color indexes in the color table that are actually 
          * used by the bitmap. If this value is zero, the bitmap uses the maximum number of colors
          * corresponding to the value of the biBitCount member for the compression mode
          * specified by biCompression. If biClrUsed is nonzero and the biBitCount member
          * is less than 16, the biClrUsed member specifies the actual number of colors
          * the graphics engine or device driver accesses. If biBitCount is 16 or
          * greater, the biClrUsed member specifies the size of the color table used to
          * optimize performance of the system color palettes. If biBitCount equals 16 or
          * 32, the optimal color palette starts immediately following the three DWORD
          * masks. If the bitmap is a packed bitmap (a bitmap in which the bitmap array
          * immediately follows the BITMAPINFO header and is referenced by a single
          * pointer), the biClrUsed member must be either zero or the actual size of the
          * color table. 
          * 
          * biClrImportant specifies the number of color indexes required
          * for displaying the bitmap. If this value is zero, all colors are required.
          * 
          * The BITMAPINFO structure combines the BITMAPINFOHEADER structure and a color
          * table to provide a complete definition of the dimensions and colors of a DIB.
          *
          * The code below reads the BMP file header section and converts the values to big endian
          */
         bmpInfoHeader_biSize = swapInt(in.readInt());                 //DWORD
         bmpInfoHeader_biWidth = swapInt(in.readInt());                //LONG
         bmpInfoHeader_biHeight = swapInt(in.readInt());               //LONG
         bmpInfoHeader_biPlanes = swapShort(in.readUnsignedShort());   //WORD
         bmpInfoHeader_biBitCount = swapShort(in.readUnsignedShort()); //WORD
         bmpInfoHeader_biCompression = swapInt(in.readInt());          //DWORD
         bmpInfoHeader_biSizeImage = swapInt(in.readInt());            //DWORD
         bmpInfoHeader_biXPelsPerMeter = swapInt(in.readInt());        //LONG
         bmpInfoHeader_biYPelsPerMeter = swapInt(in.readInt());        //LONG
         bmpInfoHeader_biClrUsed = swapInt(in.readInt());              //DWORD
         bmpInfoHeader_biClrImportant = swapInt(in.readInt());         //DWORD

         //When printing to and reading from pel files, scale each pel based on the bitmap's size per pel
         if (bmpInfoHeader_biBitCount == PEL_SIZE_WITH_RESERVED) //ignore reserved byte when scaling
            scaleFactor = Math.pow(2, SCALE_FACTOR_32_BIT);
         else //No need to worry about reserved byte if pels are not 32 bits each
            scaleFactor = Math.pow(2, bmpInfoHeader_biBitCount);

         /*
          * Since we use the height to crate arrays, it cannot have a negative a value.
          * If the height field is less than zero, then make it positive and set the topDownDIB flag to TRUE
          * so we know that the image is stored on disc upsidedown (which means it is actually rightside up).
          */
         if (bmpInfoHeader_biHeight < 0) 
         {
            topDownDIB = true;
            bmpInfoHeader_biHeight = -bmpInfoHeader_biHeight;
         }

         /*
         * Now for the color table. For true color images, there isn't one.
         */
         switch (bmpInfoHeader_biBitCount) //Determine the number of colors in the default color table
         {
            case 1:
               numberOfColors = 2;
               break;
               
            case 2:
               numberOfColors = 4;
               break;

            case 4:
               numberOfColors = 16;
               break;

            case 8:
               numberOfColors = 256;
               break;

            default:
               numberOfColors = 0; //no color table
         }

         if (bmpInfoHeader_biClrUsed > 0)
            numberOfColors = bmpInfoHeader_biClrUsed;

         for (i = 0; i < numberOfColors; ++i) //Read in the color table (or not if numberOfColors is zero)
         {
            rgbQuad_rgbBlue = in.readUnsignedByte(); //lowest byte in the color
            rgbQuad_rgbGreen = in.readUnsignedByte();
            rgbQuad_rgbRed = in.readUnsignedByte();  //highest byte in the color
            rgbQuad_rgbReserved = in.readUnsignedByte();
   
            /*
             * Build the color from the RGB values. Since we declared the rgbQuad values to
             * be int, we can shift and then OR the values to build up the color. 
             * Since we are reading one byte at a time, there are no "endian" issues.
             */
            colorPallet[i] = (rgbQuad_rgbRed << 16) | (rgbQuad_rgbGreen << 8) | rgbQuad_rgbBlue;
         } //for (i = 0; i < numberOfColors; ++i)

         /*
          * Now for the fun part. We need to read in the rest of the bitmap, but how we
          * interpret the values depends on the color depth:
          *
          * numberOfColors == 2:   Each bit is a pel, so there are 8 pels per byte. The
          *                        Color Table has only two values for "black" and "white" 
          *
          * numberOfColors == 4:   Each pair of bits is a pel, so there are 4 pels per byte. 
          *                        The Color Table has only four values 
          *
          * numberOfColors == 16:  Each nibble (4 bits) is a pel, so there are 2 pels per byte. 
          *                        The Color Table has 16 entries. 
          *
          * numberOfColors == 256: Each byte is a pel and the value maps into the 256 byte Color Table.
          *
          * Any other value is read in as "true" color.
          *
          * The BMP image is stored from bottom to top, meaning that the first scan line
          * is the last scan line in the image.
          *
          * The rest is the bitmap. Use the height and width information to read it in.
          * And as I mentioned before.... In the 32-bit format, each pixel in the image
          * is represented by a series of four bytes of RGB stored as xBRG, where the 'x'
          * is an unused byte. For ALL image types each scan line is padded to an even
          * 4-byte boundary.
          */

         imageArray = new int[bmpInfoHeader_biHeight][bmpInfoHeader_biWidth]; //Create the array for the pels
         
         switch (bmpInfoHeader_biBitCount) 
         {
            case 1: //each bit is a color, so there are 8 pels per byte. 
               /*
                * Each byte read in is 8 columns, so we need to break them out. We also have to
                * deal with the case where the image width is not an integer multiple of 8, in
                * which case we will have bits from part of the remaining byte. Each color is 1
                * bit which is masked with 0x01. The screen ordering of the pels is High-Bit to
                * Low-Bit, so the most significant element is first in the array of pels.
                */
               iBytesPerRow = bmpInfoHeader_biWidth / 8;
               iTrailingBits = bmpInfoHeader_biWidth % 8;

               iDeadBytes = iBytesPerRow;

               if (iTrailingBits > 0)
                  ++iDeadBytes;

               iDeadBytes = (4 - iDeadBytes % 4) % 4;

               for (int row = 0; row < bmpInfoHeader_biHeight; ++row) //read over the rows
               {
                  if (topDownDIB)
                     i = row;
                  else
                     i = bmpInfoHeader_biHeight - 1 - row;

                  for (j = 0; j < iBytesPerRow; ++j) 
                  {
                     iByteVal = in.readUnsignedByte();

                     for (k = 0; k < 8; ++k) //Get 8 pels from the one byte
                     {
                        iColumn = j * 8 + k;
                        pel = colorPallet[(iByteVal >> (7 - k)) & 0x01];

                        imageArray[i][iColumn] = pel;
                     }
                  }
                  if (iTrailingBits > 0) //pick up trailing bits for images that are not mod 8 columns wide
                  {
                     iByteVal = in.readUnsignedByte();

                     for (k = 0; k < iTrailingBits; ++k) 
                     {
                        iColumn = iBytesPerRow * 8 + k;
                        pel = colorPallet[(iByteVal >> (7 - k)) & 0x01];

                        imageArray[i][iColumn] = pel;
                     }
                  }

                  for (j = 0; j < iDeadBytes; ++j)
                     in.readUnsignedByte(); //Now read in the "dead bytes" to pad to a 4 byte boundary
               } //for (int row = 0; row < bmpInfoHeader_biHeight; ++row)
               break;

            case 2: //4 colors, Each byte is 4 pels (2 bits each). Not tested.
               /*
                * Each byte read in is 4 columns, so we need to break them out. We also have to
                * deal with the case where the image width is not an integer multiple of 4, in
                * which case we will have from 2 to 6 bits of the remaining byte. Each color is
                * 2 bits which is masked with 0x03. The screen ordering of the pels is
                * High-Half-Nibble to Low-Half-Nibble, so the most significant element is first
                * in the array of pels.
                */
               iBytesPerRow = bmpInfoHeader_biWidth / 4;
               iTrailingBits = bmpInfoHeader_biWidth % 4; //0, 1, 2 or 3

               iDeadBytes = iBytesPerRow;

               if (iTrailingBits > 0)
                  ++iDeadBytes;

               iDeadBytes = (4 - iDeadBytes % 4) % 4;

               for (int row = 0; row < bmpInfoHeader_biHeight; ++row) //Read over the rows
               {
                  if (topDownDIB)
                     i = row;
                  else
                     i = bmpInfoHeader_biHeight - 1 - row;

                  for (j = 0; j < iBytesPerRow; ++j) 
                  {
                     iByteVal = in.readUnsignedByte();

                     for (k = 0; k < 4; ++k) //Get 4 pels from one byte
                     {
                        iColumn = j * 4 + k;

                        //shift 2 bits at a time and reverse order
                        pel = colorPallet[(iByteVal >> ((3 - k) * 2)) & 0x03];

                        imageArray[i][iColumn] = pel;
                     }
                  }
                  if (iTrailingBits > 0) //pick up trailing nibble for images that are not mod 2 columns wide
                  {
                     iByteVal = in.readUnsignedByte();

                     for (k = 0; k < iTrailingBits; ++k) 
                     {
                        iColumn = iBytesPerRow * 4 + k;
                        pel = colorPallet[(iByteVal >> ((3 - k) * 2)) & 0x03];

                        imageArray[i][iColumn] = pel;
                     }
                  }

                  for (j = 0; j < iDeadBytes; ++j)
                     in.readUnsignedByte(); //Now read in the "dead bytes" to pad to a 4 byte boundary
               } //for (int row = 0; row < bmpInfoHeader_biHeight; ++row)
               break;

            case 4: //16 colors, Each byte is two pels.
               /*
                * Each byte read in is 2 columns, so we need to break them out. We also have to
                * deal with the case where the image width is not an integer multiple of 2, in
                * which case we will have one nibble from part of the remaining byte. We then
                * read in the dead bytes so that each scan line is a multiple of 4 bytes. Each
                * color is a nibble (4 bits) which is masked with 0x0F. The screen ordering of
                * the pels is High-Nibble Low-Nibble, so the most significant element is first
                * in the array of pels.
                */
               iPelsPerRow = bmpInfoHeader_biWidth;
               iBytesPerRow = iPelsPerRow / 2;
               iTrailingBits = iPelsPerRow % 2; //Will either be 0 or 1

               iDeadBytes = iBytesPerRow;

               if (iTrailingBits > 0)
                  ++iDeadBytes;

               iDeadBytes = (4 - iDeadBytes % 4) % 4;

               for (int row = 0; row < bmpInfoHeader_biHeight; ++row) //read over the rows
               {
                  if (topDownDIB)
                     i = row;
                  else
                     i = bmpInfoHeader_biHeight - 1 - row;

                  for (j = 0; j < iBytesPerRow; ++j) 
                  {
                     iByteVal = in.readUnsignedByte();

                     for (k = 0; k < 2; ++k) //Two pels per byte
                     {
                        iColumn = j * 2 + k; //1 - k needs to have High, Low nibble ordering for the image.
                        pel = colorPallet[(iByteVal >> ((1 - k) * 4)) & 0x0F]; //shift 4 bits at a time

                        imageArray[i][iColumn] = pel;
                     }
                  }

                  if (iTrailingBits > 0) //pick up trailing nibble for images that are not mod 2 columns wide
                  {
                     iByteVal = in.readUnsignedByte();

                     iColumn = iBytesPerRow * 2;
                     pel = colorPallet[(iByteVal >> 4) & 0x0F]; //The High nibble is the last remaining pel

                     imageArray[i][iColumn] = pel;
                  }

                  for (j = 0; j < iDeadBytes; ++j)
                     in.readUnsignedByte(); //Now read in the "dead bytes" to pad to a 4 byte boundary
               } //for (i = bmpInfoHeader_biHeight - 1; i >= 0; --i)
               break;

            case 8: //1 byte, 1 pel
               /*
                * Each byte read in is 1 column. We then read in the dead bytes so that each
                * scan line is a multiple of 4 bytes.
                */
               iPelsPerRow = bmpInfoHeader_biWidth;
               iDeadBytes = (4 - iPelsPerRow % 4) % 4;

               for (int row = 0; row < bmpInfoHeader_biHeight; ++row) //read over the rows
               {
                  if (topDownDIB)
                     i = row;
                  else
                     i = bmpInfoHeader_biHeight - 1 - row;

                  for (j = 0; j < iPelsPerRow; ++j) //j is now just the column counter
                  {
                     iByteVal = in.readUnsignedByte();
                     pel = colorPallet[iByteVal];
                     imageArray[i][j] = pel;
                  }

                  for (j = 0; j < iDeadBytes; ++j)
                     in.readUnsignedByte(); //Now read in the "dead bytes" to pad to a 4 byte boundary
               } //for (int row = 0; row < bmpInfoHeader_biHeight; ++row)
               break;

            case 16: //Not likely to work (format is not internally consistent), not tested.
               /*
                * Each two bytes read in is 1 column. Each color is 5 bits in the 2 byte word
                * value, so we shift 5 bits and then mask them off with 0x1F which is %11111 in
                * binary. We then read in the dead bytes so that each scan line is a multiple
                * of 4 bytes.
                */
               iPelsPerRow = bmpInfoHeader_biWidth;
               iDeadBytes = (4 - iPelsPerRow % 4) % 4;

               for (int row = 0; row < bmpInfoHeader_biHeight; ++row) //read over the rows
               {
                  if (topDownDIB)
                     i = row;
                  else
                     i = bmpInfoHeader_biHeight - 1 - row;

                  for (j = 0; j < iPelsPerRow; ++j) //j is now just the column counter
                  {
                     pel = swapShort(in.readUnsignedShort()); //Need to deal with little endian values

                     rgbQuad_rgbBlue = pel & 0x1F;
                     rgbQuad_rgbGreen = (pel >> 5) & 0x1F;
                     rgbQuad_rgbRed = (pel >> 10) & 0x1F;

                     pel = (rgbQuad_rgbRed << 16) | (rgbQuad_rgbGreen << 8) | rgbQuad_rgbBlue;
                     imageArray[i][j] = pel;
                  }

                  for (j = 0; j < iDeadBytes; ++j)
                     in.readUnsignedByte(); //Now read in the "dead bytes" to pad to a 4 byte boundary
               } //for (i = bmpInfoHeader_biHeight - 1; i >= 0; --i)
               break;

            case 24:
               /*
                * Each three bytes read in is 1 column. Each scan line is padded to by a
                * multiple of 4 bytes. The disk image has only 3 however.
                */
               iPelsPerRow = bmpInfoHeader_biWidth;
               iDeadBytes = (4 - (iPelsPerRow * 3) % 4) % 4;

               for (int row = 0; row < bmpInfoHeader_biHeight; ++row) //read over the rows
               {
                  if (topDownDIB)
                     i = row;
                  else
                     i = bmpInfoHeader_biHeight - 1 - row;

                  for (j = 0; j < iPelsPerRow; ++j) //j is now just the column counter
                  {
                     rgbQuad_rgbBlue = in.readUnsignedByte();
                     rgbQuad_rgbGreen = in.readUnsignedByte();
                     rgbQuad_rgbRed = in.readUnsignedByte();

                     pel = (rgbQuad_rgbRed << 16) | (rgbQuad_rgbGreen << 8) | rgbQuad_rgbBlue;
                     imageArray[i][j] = pel;
                  }

                  for (j = 0; j < iDeadBytes; ++j)
                     in.readUnsignedByte(); //Now read in the "dead bytes" to pad to a 4 byte boundary
               } //for (int row = 0; row < bmpInfoHeader_biHeight; ++row)
               break;

            case 32:
               /*
                * Each four bytes read in is 1 column. The number of bytes per line will always
                * be a multiple of 4, so there are no dead bytes.
                */
               iPelsPerRow = bmpInfoHeader_biWidth;

               for (int row = 0; row < bmpInfoHeader_biHeight; ++row) //read over the rows
               {
                  if (topDownDIB)
                     i = row;
                  else
                     i = bmpInfoHeader_biHeight - 1 - row;

                  for (j = 0; j < iPelsPerRow; ++j) //j is now just the column counter
                  {
                     rgbQuad_rgbBlue = in.readUnsignedByte();
                     rgbQuad_rgbGreen = in.readUnsignedByte();
                     rgbQuad_rgbRed = in.readUnsignedByte();
                     rgbQuad_rgbReserved = in.readUnsignedByte();

                     pel = (rgbQuad_rgbReserved << 24) | (rgbQuad_rgbRed << 16) | (rgbQuad_rgbGreen << 8)
                           | rgbQuad_rgbBlue;
                     
                     imageArray[i][j] = pel;
                  } //for (j = 0; j < iPelsPerRow; ++j)
               } //for (int row = 0; row < bmpInfoHeader_biHeight; ++row)
               break;

            default: //Unexpected bits per pel value from BMP header
               System.out.printf("This error should not occur - 1!\n");
         } //switch (bmpInfoHeader_biBitCount)

         in.close();
         fstream.close();
      } //try
      catch (Exception e) 
      {
         e.printStackTrace();
      }
   } //public void readBitmapFromFile(String inFileName)

   /**
    * Writes out the pel values of the input BMP to a text file, with each row of pels on its own line
    * and each pel value seperated by a space. Pels are scaled down to a range of 0 to 1,
    * and are scaled back up if the pel file is used with this class' readPelsFromFile method.
    *
    * The pels are scaled by '-scaleFactor' instead of 'scaleFactor' 
    * to make the values in the pel file positive.
    *
    * @param outPelFileName
    *       The file name or file path for the file (existing or to be created) 
    *       for which file the scaled pel values should be placed in
    *
    * @param grayScale
    *       True if the pel file should contain pels for the gray scale version of the color BMP the
    *       BMPUtil object contains, false if the color pels should be used.
    */
   public void printPelsToFile(String outPelFileName, boolean grayScale)
   {
      try
      {
         PrintStream pelPrinter = new PrintStream(outPelFileName);

         for (int[] pelRow : imageArray)
         {
            for (int pel : pelRow)
            {
               if (grayScale)
                  pel = colorToGrayscale(pel);

               pelPrinter.print((double)pel / scaleFactor);
               pelPrinter.print(" ");
            }

            pelPrinter.println(); //New line for new pel row
         }

         pelPrinter.close();
      } //try
      catch (FileNotFoundException e)
      {
         System.out.println("Pel File could not be created" + e.getStackTrace());
      }
   } //public void printPelsToFile(String outPelFileName)

   /**
    * Instantiates and fills the imageArray with pels from the pel file at inFileName. After
    * this method completes execution, imageArray will be instantiated with the pel values from
    * the pel file and an output BMP can be generated.
    *
    * The pels are scaled by '-scaleFactor' instead of 'scaleFactor' to account for the use of
    * '-scaleFactor' instead of 'scaleFactor' when printing the pels to the pel file.
    *
    * @param inFileName 
    *       The name of the file containing the pel file created from a BMP file by a BMPUtil object
    */
   public void readPelsFromFile(String inFileName)
   {
      try
      {
         Scanner s = new Scanner(new File(inFileName));

         for (int pelRow = 0; pelRow < bmpInfoHeader_biHeight; pelRow++)
         {
            for (int pelIndex = 0; pelIndex < bmpInfoHeader_biWidth; pelIndex++)
            {
               double scaledPel = s.nextDouble();
               int rawPel = Math.round((float)(scaledPel * -scaleFactor));

               imageArray[pelRow][pelIndex] = rawPel; 
            }
         }
      } //try
      catch (Exception e)
      {
         e.printStackTrace();
      }
   } //public void readPelsFromFile(String inFileName)

   /**
    * Writes out the true color bitmap to a BMP file, using the stored pel values and BMP header data.
    *
    * @param outBMPName
    *       The file name or file path for the file (existing or to be created) 
    *       for the output BMP file, constructed from the same pels as the input BMP,
    *       to be placed at.
    */
   public void printBitmapToFile(String outBMPName)
   {
      try 
      {
         iDeadBytes = (4 - (bmpInfoHeader_biWidth * 3) % 4) % 4;

         bmpInfoHeader_biSizeImage = (bmpInfoHeader_biWidth * 3 + iDeadBytes) * bmpInfoHeader_biHeight;
         bmpFileHeader_bfOffBits = DEFAULT_OFFSET;
         bmpFileHeader_bfSize = bmpInfoHeader_biSizeImage + bmpFileHeader_bfOffBits;
         bmpInfoHeader_biBitCount = DEFAULT_BITS_PER_PEL;
         bmpInfoHeader_biCompression = BI_RGB;
         bmpInfoHeader_biClrUsed = TRUE_COLOR;
         bmpInfoHeader_biClrImportant = TRUE_COLOR;

         FileOutputStream bmpFileStream = new FileOutputStream(outBMPName);
         DataOutputStream bmpDataStream = new DataOutputStream(bmpFileStream);

         //BITMAPFILEHEADER
         bmpDataStream.writeShort(swapShort(bmpFileHeader_bfType));      //WORD
         bmpDataStream.writeInt(swapInt(bmpFileHeader_bfSize));          //DWORD
         bmpDataStream.writeShort(swapShort(bmpFileHeader_bfReserved1)); //WORD
         bmpDataStream.writeShort(swapShort(bmpFileHeader_bfReserved2)); //WORD
         bmpDataStream.writeInt(swapInt(bmpFileHeader_bfOffBits));       //DWORD

         //BITMAPINFOHEADER
         bmpDataStream.writeInt(swapInt(bmpInfoHeader_biSize));          //DWORD
         bmpDataStream.writeInt(swapInt(bmpInfoHeader_biWidth));         //LONG
         bmpDataStream.writeInt(swapInt(bmpInfoHeader_biHeight));        //LONG
         bmpDataStream.writeShort(swapShort(bmpInfoHeader_biPlanes));    //WORD
         bmpDataStream.writeShort(swapShort(bmpInfoHeader_biBitCount));  //WORD
         bmpDataStream.writeInt(swapInt(bmpInfoHeader_biCompression));   //DWORD
         bmpDataStream.writeInt(swapInt(bmpInfoHeader_biSizeImage));     //DWORD
         bmpDataStream.writeInt(swapInt(bmpInfoHeader_biXPelsPerMeter)); //LONG
         bmpDataStream.writeInt(swapInt(bmpInfoHeader_biYPelsPerMeter)); //LONG
         bmpDataStream.writeInt(swapInt(bmpInfoHeader_biClrUsed));       //DWORD
         bmpDataStream.writeInt(swapInt(bmpInfoHeader_biClrImportant));  //DWORD

         //there is no color table for this true color image, so write out the pels

         rgbQuad_rgbReserved = DEFAULT_RESERVED_VALUE;

         for (i = bmpInfoHeader_biHeight - 1; i >= 0; --i) //write over the rows (in inverted format)
         {
            for (j = 0; j < bmpInfoHeader_biWidth; ++j) //and the columns
            {
               pel = imageArray[i][j];

               rgbQuad_rgbBlue = pel & 0x00FF;
               rgbQuad_rgbGreen = (pel >> 8) & 0x00FF;
               rgbQuad_rgbRed = (pel >> 16) & 0x00FF;

               bmpDataStream.writeByte(rgbQuad_rgbBlue); //lowest byte in the color
               bmpDataStream.writeByte(rgbQuad_rgbGreen);
               bmpDataStream.writeByte(rgbQuad_rgbRed);  //highest byte in the color
               
               if (bmpInfoHeader_biBitCount == PEL_SIZE_WITH_RESERVED)
                  bmpDataStream.writeByte(rgbQuad_rgbReserved);
            } //for (j = 0; j < bmpInfoHeader_biWidth; ++j)

            if (bmpInfoHeader_biBitCount != PEL_SIZE_WITH_RESERVED) //Has dead bytes and not reserved bytes
            {
               for (j = 0; j < iDeadBytes; ++j)
                  bmpDataStream.writeByte(DEFAULT_DEAD_BYTE); //Write out "dead bytes" to pad a 4 byte boundary
            }
         } //for (i = bmpInfoHeader_biHeight - 1; i >= 0; --i)

         bmpDataStream.close();
         bmpFileStream.close();
      } //try
      catch (Exception e) 
      {
         e.printStackTrace();
      }
   } //public void printBitmapToFiles(String outBMPName, String outPelFileName)

   /**
    * Main method, which creates a BMPUtil object over a BMP file and print's the input BMP's pels to a file.
    * Useful for making a pel file which can be used in the training section of a Model's config file 
    * (see 'TemplateConfigTestFiles.txt' in the models folder for an example of using a file 
    * as inputs/outputs for a test case). The pel files are automatically converted to gray scale.
    * 
    * @param args
    *       The first argument must specify the file name or file path of the input BMP. 
    *       If the file name/path is invalid or the file at the name/path does not exist, 
    *       a FileNotFoundException will be thrown.
    *       The second argument must specify the file name or path of the file where the pel values from the 
    *       input BMP should be placed. This file will be automatically created if it does not exist, 
    *       but a FileNotFoundException will be thrown if the name/path is invalid 
    *       or the file cannot be created for whatever reason. 
    */
   public static void main(String[] args) 
   {
      String inFileName = args[0];
      String outPelFileName = args[1];

      BMPUtil dd = new BMPUtil(inFileName);
      dd.printPelsToFile(outPelFileName, true);
   }
} //public class BMPUtil