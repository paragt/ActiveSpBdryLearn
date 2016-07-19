

#include "IterativeLearn.h"

// using namespace NeuroProof;


void IterativeLearn::update_clfr_name(string &clfr_name, size_t trnsz)
{
  
    size_t found = clfr_name.find("trnsz");
    size_t end = clfr_name.find("_",found+1);
    string newstr = "trnsz"; 
    char szstr[255];
    sprintf(szstr,"%d",trnsz);//(trnsz, szstr,10);
    newstr += szstr;
    
    clfr_name.replace(clfr_name.begin()+found,clfr_name.begin()+end, newstr);
}








void IterativeLearn::edgelist_from_index(std::vector<unsigned int>& new_idx,
					      std::vector< std::pair<unsigned int, unsigned int> >& elist)
{
  
    elist.clear();
    elist.resize(new_idx.size());
    for(size_t ii=0; ii < new_idx.size(); ii++){
	unsigned int idx = new_idx[ii];
	elist[ii] = edgelist[idx];
    }
}

void IterativeLearn::save_classifier(string &clfr_name){
    std::vector<unsigned int> ignore_list;
    std::set<unsigned int>::iterator iit = ignore_set.begin();
    for(; iit != ignore_set.end(); iit++)
	ignore_list.push_back((*iit));
    
//     eclfr->set_ignore_featlist(ignore_list);
    
//     eclfr->save_classifier(clfr_name.c_str());
    
// // // //     eclfr->set_ignore_featlist(ignore_list);
// // // //     
    eclfr->save_classifier(clfr_name.c_str());
}




// **************************************************************************************************
void IterativeLearn::read_all_edge_features(std::vector< std::vector<double> >& all_features,
					      std::vector<int>& all_labels){
    
  
    all_features.clear();
    all_labels.clear();
    
    FILE* fp = fopen(feature_filename.c_str(),"rt");
    unsigned int row, col, nch;
    fscanf(fp,"%u %u %u\n",&row, &col, &nch);
    nfeat_channels = nch;
    std::vector<double> feat1(col,0);
    double ff;
    int ll;
    for(size_t  i = 0; i<row; i++){
	for(size_t j=0 ; j<col; j++){
	    fscanf(fp, "%lf ", &ff);
	    feat1[j] = ff;
	}
	fscanf(fp, "%d ", &ll);
	
	ll = (ll<0? NEGATIVE : POSITIVE);
	
	all_features.push_back(feat1);
	all_labels.push_back(ll);
    }
	  
    
    fclose(fp);

						
//     unsigned int tmp_ignore[] = {0, 55, 110, 165}; 
// 	tmp_ignore[ff] = tmp_ignore[ff-1]+ (1 + nfeat_channels*4 + nfeat_channels *5); 
    
//     std::vector<unsigned int> ignore_list(tmp_ignore, tmp_ignore + sizeof(tmp_ignore)/sizeof(unsigned int));
    
    find_useless_features(all_features);
    printf("ignore features:");
    std::set<unsigned int>::iterator iit = ignore_set.begin();
    for(; iit != ignore_set.end(); iit++)
	printf("%u ", *iit);
    printf("\n");
    
}

void IterativeLearn::find_useless_features(std::vector< std::vector<double> >& all_features)
{
    
    unsigned int tmp_ignore[4];
    tmp_ignore[0] = 0;
    for(size_t ff=0 ;ff<4; ff++)
	ignore_set.insert(ff*(1 + nfeat_channels*4 + nfeat_channels *5));
    
    
    unsigned int nfeat = all_features[0].size();
    unsigned int nsamples = all_features.size();
    for(size_t ff=0; ff< nfeat; ff++){
      
	double fmean = 0;
	for(size_t ii=0; ii < nsamples; ii++)
	    fmean += all_features[ii][ff];
	fmean /= nsamples;
	
	double fvar = 0;
	for(size_t ii=0; ii < nsamples; ii++)
	    fvar += (all_features[ii][ff] - fmean)*(all_features[ii][ff] - fmean);
	fvar /= nsamples;
	
	double fstdev = sqrt(fvar);
	
	if (fstdev < 0.001){
	    ignore_set.insert(ff);
	}
    }
    ignore_set.insert(nfeat);
    
// // // //     unsigned int nfeat2 = nfeat - ignore_set.size();
// // // //     std:vector <double> newfeat;
// // // //     for(size_t ii=0; ii < nsamples; ii++){
// // // // 	newfeat.clear();
// // // // 	unsigned int next = 0,ff =0;
// // // // 	std::set<unsigned int>::iterator iiter = ignore_set.begin();
// // // // 	while(ff<nfeat){
// // // // 	    for(; (ff< (*iiter)) && (ff<nfeat); ff++)
// // // // 		newfeat.push_back(all_features[ii][ff]);
// // // // 	    if((ff<nfeat) && (ff == (*iiter))){
// // // // 		ff++;
// // // // 		if (iiter!=ignore_set.end()) 
// // // // 		  iiter++;
// // // // 	    }
// // // // 	}
// // // // 	all_features[ii] = newfeat;
// // // //     }
    
    
}



/*void IterativeLearn::compute_all_edge_features(std::vector< std::vector<double> >& all_features,
					      std::vector<int>& all_labels){

    
    int count=0; 	
    for (Rag_t::edges_iterator iter = rag->edges_begin(); iter != rag->edges_end(); ++iter) {

        if ( (!(*iter)->is_preserve()) && (!(*iter)->is_false_edge()) ) {
	    RagEdge_t* rag_edge = *iter; 	

            RagNode_t* rag_node1 = rag_edge->get_node1();
            RagNode_t* rag_node2 = rag_edge->get_node2();
            Node_t node1 = rag_node1->get_node_id(); 
            Node_t node2 = rag_node2->get_node_id(); 

// 	    int edge_label = stack->decide_edge_label(rag_node1, rag_node2);
	    int edge_label;
	    if (stack->get_gt_labelvol()==NULL){
		edge_label=2;
	    }
	    else{
		edge_label= stack->find_edge_label(node1, node2);
		
		if (use_mito && (stack->is_mito(node1) || 
				stack->is_mito(node2))) {
		    edge_label = 1; 
		}

	    }
	
 	    unsigned long long node1sz = rag_node1->get_size();	
	    unsigned long long node2sz = rag_node2->get_size();	

	    if ( edge_label ){	
		std::vector<double> feature;
		feature_mgr->compute_all_features(rag_edge,feature);

		all_features.push_back(feature);
		all_labels.push_back(edge_label);

		edgelist.push_back(std::make_pair(node1,node2));
// 		feature.push_back(edge_label);
// 		all_featuresu.insert(feature);

	    }	
	    else 
		int checkp = 1;	

        }
    }

}*/



void IterativeLearn::evaluate_accuracy(std::vector< std::vector<double> >& test_features,
			       std::vector<int>& test_labels, double thd )
{
    size_t nexamples2tst = test_features.size();
    size_t nexamples_p = 0;
    size_t nexamples_n = 0;
    
    
    
    double corr_p=0, corr_n=0, fp=0, fn= 0 ;
    for(int ecount=0;ecount< test_features.size(); ecount++){
// 	double predp = feature_mgr->get_classifier()->predict(test_features[ecount]);
	double predp = eclfr->predict(test_features[ecount]);
	int predl = (predp > thd)? POSITIVE:NEGATIVE;	
	int actuall = test_labels[ecount];
	
	actuall==1 ? nexamples_p++ : nexamples_n++;
	
	corr_p += ((predl== POSITIVE && actuall ==POSITIVE) ? 1: 0);
	corr_n += ((predl== NEGATIVE &&  actuall ==NEGATIVE) ? 1: 0);
	fp += ((predl== POSITIVE &&  actuall == NEGATIVE) ? 1: 0);
	fn += ((predl== NEGATIVE &&  actuall == POSITIVE) ? 1: 0);
	
	//err+= ((predl== subset_labels[ecount])?0:1);	
    }
    //printf("accuracy = %.3f\n",100*(1 - err/subset_labels.size()));	
    printf("total tst samples= %u\n",nexamples2tst);
    printf("correct p = %.1f\n", corr_p*100.0/nexamples2tst);
    printf("correct n = %.1f\n", corr_n*100.0/nexamples2tst);
    printf("false p = %.1f\n", fp*100.0/nexamples2tst);
    printf("false n = %.1f\n", fn*100.0/nexamples2tst);

}
void IterativeLearn::evaluate_accuracy(std::vector<int>& labels, std::vector<double>& predicted_vals, double thd)
{
    size_t nexamples2tst = labels.size();
    size_t nexamples_p = 0;
    size_t nexamples_n = 0;
    
//     double thd = 0.3 ;
    
    double corr_p=0, corr_n=0, fp=0, fn= 0, undecided=0 ;
    for(int ecount=0;ecount< labels.size(); ecount++){
	double predp = predicted_vals[ecount];
	int predl = (predp > thd)? POSITIVE :NEGATIVE;
	if (!(predp > thd) &&  !(predp<thd)){
	  predl=0;
	  undecided++;
	}
	int actuall = labels[ecount];
	
	actuall==1 ? nexamples_p++ : nexamples_n++;
	
	corr_p += ((predl== POSITIVE && actuall == POSITIVE) ? 1: 0);
	corr_n += ((predl== NEGATIVE &&  actuall ==NEGATIVE) ? 1: 0);
	fp += ((predl== POSITIVE &&  actuall == NEGATIVE) ? 1: 0);
	fn += ((predl== NEGATIVE &&  actuall == POSITIVE) ? 1: 0);
	
	//err+= ((predl== subset_labels[ecount])?0:1);	
    }
    //printf("accuracy = %.3f\n",100*(1 - err/subset_labels.size()));	
    printf("total tst samples= %u\n",nexamples2tst);
    printf("correct p = %.1f\n", corr_p*100.0/nexamples2tst);
    printf("correct n = %.1f\n", corr_n*100.0/nexamples2tst);
    printf("false p = %.1f\n", fp*100.0/nexamples2tst);
    printf("false n = %.1f\n", fn*100.0/nexamples2tst);
    printf("undecided samples= %.2f\n",undecided*100.0/nexamples2tst);

}


