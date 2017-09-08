# 3D Box class
# Author: Mustafa Haddara
# This class represents a rectangle as (x,y,z) coordinate representing the 
# front-facing top-left corner, and a (w,h,d) tuple representing the width,
# height, depth
## @mainpage Mainpage
#
##
# @file Box3D.py
# @title Box3D
# @author Mustafa Haddara
# @date 1/10/2016
# @brief This class represents a rectangle. 
# @details This class represents a rectangle as (x,y,z) coordinate representing the 
# front-facing top-left corner, and a (w,h,d) tuple representing the width, height and depth. 	
# @code
#	b = Box3D( (1,1,1), (2,2,2) )
#    print b.getSurfaceArea()
# @endcode		
# @todo Add test cases for the functions.
class Box3D(object):
    ## Faces enums
	#
    FRONT = 0 ## enum value FRONT#
    TOP = 1    ## enum value TOP#
    SIDE = 2   ## enum value SIDE.#
	 
	## @brief Constructor for Box3D
	# @details Constructor accepts two parameters for the top left corner and size dimensions.
    # @param corner (x,y,z) tuple for the top left corner.
    # @param size (w,h,d) tuple containing the width, height and depth.
    # Constructor accepts 2 parameters
    # corner: (x,y,z) tuple for the top left corner
    # size: (w,h,d) tuple containing the width and height
    def __init__(self, corner, size):
        if len(corner) != 3 or len(size) != 3:
            raise ValueError("Invalid argument size")
        # double underscore (__) before the identifier is python convention
        # for private variables
        self.__frontTopLeftCorner = corner
        self.__dimensions = size
	
	##  @brief returns the front top left corner tuple.
	#  @return frontTopLeftCorner
     
	 def getFrontTopLeftCorner(self):
        return self.__frontTopLeftCorner
	
	 ## @brief sets the front top left corner.
	 # @param newCorner: (x,y,z) tuple for the front top left corner
     
    def setFrontTopLeftCorner(self, newCorner):
        self.__frontTopLeftCorner = newCorner
		
	##  @brief gets width, height and depth.
	#  @return dimensions
      
    def getDimensions(self):
        return self.__dimensions
	 ##@brief gets width, height and depth.
	 # @param dimensions
      
	  def setDimensions(self, newSize):
        self.__dimensions = newSize
	## @brief This function sets the dimensions of the given face.
	# @author Mustafa Haddara
	# @param face Face of the box: FRONT or TOP or SIDE.
	# @return width, hight.
		 
    def getDimensionsOfFace(self, face):
        if face == Box3D.FRONT:
            width,height = self.__dimensions[0],self.__dimensions[1] # w,h
        elif face == Box3D.TOP:
            width,height = self.__dimensions[0],self.__dimensions[2] # w,d
        elif face == Box3D.SIDE:
            width,height = self.__dimensions[1],self.__dimensions[2] # h,d
        else:
            raise ValueError("Invalid `face` argument")
        return width,height
		
	## @brief This function calculates the area of the given face.
	# @author Mustafa Haddara
	# @param face Face of the box: FRONT or TOP or SIDE.
	# @return Area of the given face.
	# @date 1/10/2016
				
    def getAreaOfFace(self, face):
        width,height = self.getDimensionsOfFace(face)
        return width * height
		
	##	@brief This function calculates the perimeter of the given face.
	#	@author Mustafa Haddara
	#	@param face Face of the box: FRONT or TOP or SIDE.
	#	@return Perimeter of the given face.
	#	@date 1/10/2016
	
    def getPerimeterOfFace(self, face):
        width,height = self.getDimensionsOfFace(face)
        return 2 * (width + height)
	
	##	@brief This function calculates the volume of the rectangle.
	#	@author Mustafa Haddara
	#	@return Volume of the rectangle.
	#	@date 1/10/2016
	
    def getVolume(self):
        width,height,depth = self.__dimensions
        return width * height * depth
	
	##	@brief This function calculates the surface area of the rectangle.
	#	@author Mustafa Haddara
	#	@return Surface area of the rectangle.
	#	@date 1/10/2016
	
    def getSurfaceArea(self):
        surfaceArea = 0
        for face in (Box3D.FRONT, Box3D.TOP, Box3D.SIDE):
            surfaceArea += (2 * self.getAreaOfFace(face))
        return surfaceArea
	
	##	@brief Example of the usage.
	#	@author Mustafa Haddara
	#	@date 1/10/2016
	
	if __name__ == '__main__':
    b = Box3D( (1,1,1), (2,2,2) )
    print b.getSurfaceArea()