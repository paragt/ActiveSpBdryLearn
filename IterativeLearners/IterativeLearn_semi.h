#ifndef ITERATIVE_LEARN_SEMI
#define ITERATIVE_LEARN_SEMI

#include "IterativeLearn.h"
// #include "../SemiSupervised/weightmatrix1.h"
#include "../SemiSupervised/weightmatrix_iter.h"
#include "../SemiSupervised/kmeans.h"
// #include "../Utilities/unique_row_matrix.h"




// #define CHUNKSZ_SM 10 
// #define INITPCT_SM 0.035

#define DEGREE 1
#define KMEANS 2

// namespace NeuroProof{

 

class IterativeLearn_semi:public IterativeLearn{

    WeightMatrix_iter* wt1;
//     WeightMatrix_iter* wti;
    std::vector<unsigned int> trn_idx;
    std::vector<unsigned int> init_trn_idx;
    int initial_set_strategy;
    
    boost::thread* threadp;
//     std::vector<unsigned int> nz_degree_idx;
//     std::vector<unsigned int> all_idx;
//     std::vector<unsigned int> tst_idx;
    const double INITPCT_SM ;
    const double CHUNKSZ_SM ;
    
    double w_dist_thd ;
    
public:
    IterativeLearn_semi(string feature_filename, string pclfr_name, double pwt_mat_thd): IterativeLearn(feature_filename, pclfr_name),
				  INITPCT_SM(0.035), CHUNKSZ_SM(10){
	trn_idx.clear();
	edgelist.clear();
	initial_set_strategy = KMEANS; //KMEANS
	
	if (endswith(pclfr_name, ".h5")){
	    string nameonly = pclfr_name.substr(0, pclfr_name.find_last_of("."));	
	    if (nameonly.find("parallel") != std::string::npos)
		eclfr = new VigraRFclassifierP();
	    else
		eclfr = new VigraRFclassifier();	
	      
	}
	else if (endswith(pclfr_name, ".xml")){
	    string nameonly = pclfr_name.substr(0, pclfr_name.find_last_of("."));	
	    if (nameonly.find("parallel") != std::string::npos)
		eclfr = new OpencvRFclassifierP();	
	    else
		eclfr = new OpencvRFclassifier();	
	}
	
	w_dist_thd = pwt_mat_thd;    
	
    }
    void get_initial_edges(std::vector<unsigned int>& new_idx);
    void learn_edge_classifier(double trnsz);
    void update_new_labels(std::vector<unsigned int>& new_idx, std::vector<int>& new_lbl);
    void get_next_edge_set(size_t feat2add, std::vector<unsigned int>& new_idx);

    void prune_features(std::vector< std::vector<double> >&, std::vector<unsigned int>& ignore_idx);
    
    std::vector<unsigned int>& get_trn_idx(){ return trn_idx;};
    void get_extra_edges(std::vector<unsigned int>& ret_idx, size_t nedges);
    void compute_new_risks(std::multimap<double, unsigned int>& risks, 
			   std::map<unsigned int, double>& prop_lbl);
    
};



// }
#endif