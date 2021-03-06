#ifndef ITERATIVE_LEARN_
#define ITERATIVE_LEARN_

#include "../Dataset.h"
// #include <boost/thread.hpp>
// #include <boost/ref.hpp>
#include "../Classifier/opencvRFclassifier.h"
#include "../Classifier/opencvRFclassifierP.h"
#include "../Classifier/vigraRFclassifier.h"
#include "../Classifier/vigraRFclassifierP.h"


#include <float.h>
#include <cstdlib>


#define IVAL 500

#define POSITIVE 1
#define NEGATIVE -1


// namespace NeuroProof{


class IterativeLearn
{

protected:
//     UniqueRowFeature_Label all_featuresu;
//     std::vector<int> all_labels;  
//     std::vector< std::vector<double> > all_features;
    Dataset dtst;
    
//     BioStack* stack;
//     RagPtr rag;
//     FeatureMgrPtr feature_mgr;

    std::vector< std::vector<double> > cum_train_features;
    std::vector< int > cum_train_labels;
    std::vector< std::vector<double> > rest_features;
    std::vector< int > rest_labels;

    
    string clfr_name; 
    string feature_filename; 
    
    bool use_mito;
    
    EdgeClassifier* eclfr;
    EdgeClassifier* eclfrP;
    unsigned int nfeat_channels;
    
    
    std::vector< std::pair<unsigned int, unsigned int> > edgelist;
    std::set<unsigned int> ignore_set;
    
public:
  
    IterativeLearn(string pfeature_filename, string pclfr_name=""){
// 	stack = pstack;
// 	rag = stack->get_rag();
// 	feature_mgr = stack->get_feature_manager();
	feature_filename = pfeature_filename;
	clfr_name = pclfr_name;
	
	use_mito = true;
    };

    
    
    
    
    
    virtual void get_initial_edges(std::vector<unsigned int>& new_idx) = 0;
    virtual void update_new_labels(std::vector<unsigned int>& new_idx, 
				   std::vector<int>& new_lbl) = 0;
    virtual void get_next_edge_set(size_t feat2add, std::vector<unsigned int>& new_idx) = 0;
    
    
    virtual void learn_edge_classifier(double trnsz){};
    
    
    //void learn_edge_classifier_flat_random(double trnsz);

    void edgelist_from_index(std::vector<unsigned int>& new_idx,
			      std::vector< std::pair<unsigned int, unsigned int> >& elist);

    void read_all_edge_features(std::vector< std::vector<double> >& all_features,
					      std::vector<int>& all_labels);

    void evaluate_accuracy(std::vector< std::vector<double> >& test_features,
			       std::vector<int>& test_labels, double thd );
    void evaluate_accuracy(std::vector<int>& labels, std::vector<double>& predicted_vals, double thd);

    void update_clfr_name(string &clfr_name, size_t trnsz);

    void save_classifier(string &clfr_name);

    void find_useless_features(std::vector< std::vector<double> >& all_features);

  
    bool endswith(string filename, string extn){
      unsigned found = filename.find_last_of(".");
      string fextn = filename.substr(found);	
      if (fextn.compare(extn) == 0 )
	  return true;
      else return false;	  
    }
    
    
};





// }
#endif