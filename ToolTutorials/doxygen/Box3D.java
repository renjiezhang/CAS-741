/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
@mainpage The mainpage documentation for Box3D.
*/
/**
* @file Box3D.java
* @title Box3D
* @author Mustafa Haddara
* @date 1/10/2016
* @brief This class represents a rectangle. 
* @details This class represents a rectangle as (x,y,z) coordinate representing the 
* front-facing top-left corner, and a (w,h,d) tuple representing the width, height and depth. 	
* @code
*    Box3D b = new Box3D( new int[] {1,1,1}, new int[] {2,2,2});
*    System.out.println(b.getVolume());
*    System.out.println(b.getSurfaceArea());
* @endcode		
* @todo Add test cases for the functions.
*/
package box3d;

/* 3D Box class
 * Author: Mustafa Haddara
 *
 * This class represents a rectangle as (x,y,z) coordinate representing the 
 * front-facing top-left corner, and a (w,h,d) tuple representing the width,
 * height, depth
 */
public class Box3D {
    /** 
     * Face enums
     */
    enum Face {
        FRONT, /**< enum value FRONT. */
        TOP, /**< enum value TOP. */
        SIDE /**< enum value SIDE. */
    }
	 /** 
       * a private variable for the (x,y,z) coordinate.
	   */
    private int[] frontTopLeftCorner;
	 /** 
       * a private variable for the width,
       * height and depth.
       */
    private int[] dimensions;
    /** 
	 * @brief Constructor for Box3D
	 * @details Constructor accepts two parameters for the top left corner and size dimensions.
     * @param corner (x,y,z) tuple for the top left corner.
     * @param size (w,h,d) tuple containing the width, height and depth.
     */
    public Box3D (int[] corner, int[] size) {
        if(corner.length != 3 || size.length != 3) {
            throw new RuntimeException("Invalid argument size");
        }
        this.frontTopLeftCorner = corner;
        this.dimensions = size;
    }
	/** 
	 * @brief returns the front top left corner tuple.
	 * @return frontTopLeftCorner
     */
    public int[] getFrontTopLeftCorner() {
        return this.frontTopLeftCorner;
    }
	/** 
	 * @brief  sets the front top left corner.
	 * @param newCorner Front top left corner coordinates.
     */
    public void setFrontTopLeftCorner(int[] newCorner) {
        this.frontTopLeftCorner = newCorner;
    }
	/** 
	 * @brief  returns width, height and depth.
	 * @return dimensions
     */
    public int[] getDimensions() {
        return this.dimensions;
    }
	/** 
	 * @brief sets width, height and depth.
	 * @param newSize dimensions
     */
    public void setDimensions(int[] newSize) {
        this.dimensions = newSize;
    }

/**
* @brief This function sets the dimensions of the given face.
* @author Mustafa Haddara
* @param face Face of the box: FRONT or TOP or SIDE.
* @return dimension of the given face.
* @date 1/10/2016
*/
    public int[] getDimensionsOfFace(Face face) {
        int[] dim = new int[3];
        if (face == Face.FRONT) {
            dim = new int[] { this.dimensions[0], this.dimensions[1] };
        } else if (face == Face.TOP) {
            dim = new int[] { this.dimensions[0], this.dimensions[2] };
        } else if (face == Face.SIDE) {
            dim = new int[] { this.dimensions[1], this.dimensions[2] };
        }
        return dim;
    }
/**
* @brief This function calculates the area of the given face.
* @author Mustafa Haddara
* @param face Face of the box: FRONT or TOP or SIDE.
* @return Area of the given face.
* @date 1/10/2016
*/
    public int getAreaOfFace(Face face) {
        int[] dim = this.getDimensionsOfFace(face);
        return dim[0] * dim[1];
    }
	
/**
* @brief This function calculates the perimeter of the given face.
* @author Mustafa Haddara
* @param face Face of the box: FRONT or TOP or SIDE.
* @return Perimeter of the given face.
* @date 1/10/2016
*/
    public int getPerimeterOfFace(Face face) {
        int[] dim = this.getDimensionsOfFace(face);
        return 2 * (dim[0] + dim[1]);
    }
/**
* @brief This function calculates the volume of the rectangle.
* @author Mustafa Haddara
* @return Volume of the rectangle.
* @date 1/10/2016
*/
    public int getVolume() {
        int width  = this.dimensions[0];
        int height = this.dimensions[1];
        int depth  = this.dimensions[2];
        return width * height * depth;
    }
/**
* @brief This function calculates the surface area for the rectangle.
* @author Mustafa Haddara
* @return Surface area of the rectangle.
* @date 1/10/2016
*/
    public int getSurfaceArea() {
        int surfaceArea = 0;
        for (Face face: Face.values()) {
           surfaceArea += (2* this.getAreaOfFace(face));
        }
        return surfaceArea;
    }
/**
* @brief Example of the usage.
* @author Mustafa Haddara
* @date 1/10/2016
*/
    public static void main(String[] args) {
        Box3D b = new Box3D( new int[] {1,1,1}, new int[] {2,2,2});
        System.out.println(b.getVolume());
        System.out.println(b.getSurfaceArea());
    }
}
    

