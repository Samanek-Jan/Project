/*
 * Test class
*/
#template<tyoename T>
class T
{
   private: 
    double x;
    double y;

    T()
    {
        x = 0;
        y = 0;
    }

    ~T()
    {
        x = 1;
    }
} // End of class