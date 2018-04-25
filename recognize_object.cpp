#include <recognize_objects.h>

int
main (int argc, char** argv)
{
	int k = 6;
	double thresh = DBL_MAX;
	std::string model_dir = "data";
	std::string inputname = "input.pcd";
	std::vector<vfh_model> models;	
  	flann::Matrix<int> k_indices;
  	flann::Matrix<float> k_distances;

	if(argc<2) {
	PCL_ERROR("Usage: %s [-i <inputname>] [-m <model_dir>] [-k <neighbors>] [-t <threshold>]\n", argv[0]);
	return(-1);
	}
	
	// Parse console inputs
	pcl::console::parse_argument (argc, argv, "-i", inputname);
	pcl::console::parse_argument (argc, argv, "-m", model_dir);
	pcl::console::parse_argument (argc, argv, "-k", k);
	pcl::console::parse_argument (argc, argv, "-t", thresh);

  	vfh_model histogram;
  	std::string str = inputname;
	// Check if the query object has VFH signature, if not estimate signature and save it in a file
  	if (!checkVFH(inputname))
  	{
      pcl::PointCloud <pcl::VFHSignature308> signature;
      estimate_VFH(str, signature);
      str.replace(str.find(".pcd"), 4, "_vfh.pcd");
      pcl::io::savePCDFile (str, signature);
  	}
	// Load VFH signature of the query object
    if(!loadHist (str, histogram))
  	{
  	  pcl::console::print_error ("Cannot load test file %s\n", inputname);
  	  return (-1);
  	}
	
	// Load training data
	loadData(model_dir, models);

  	// Convert data into FLANN format
  	flann::Matrix<float> data (new float[models.size () * models[0].second.size ()], models.size (), models[0].second.size ());
	
  	for (size_t i = 0; i < data.rows; ++i)
  	  for (size_t j = 0; j < data.cols; ++j)
  	    data[i][j] = models[i].second[j];

	// Place data in FLANN K-d tree
	flann::Index<flann::ChiSquareDistance<float> > index (data, flann::LinearIndexParams ());
	index.buildIndex ();
	
	// Search for query object in the K-d tree
	nearestKSearch (index, histogram, k, k_indices, k_distances);

	// Print closest candidates on the console
	std::cout << "The closest " << k << " neighbors for " << inputname << " are: " << std::endl;
  	for (int i = 0; i < k; ++i)
    	pcl::console::print_info ("    %d - %s (%d) with a distance of: %f\n", 
        i, models.at (k_indices[0][i]).first.c_str (), k_indices[0][i], k_distances[0][i]);

	// Visualize closest candidates on the screen
	visualize(argc, argv, k, thresh, models, k_indices, k_distances);

	return 0;
}