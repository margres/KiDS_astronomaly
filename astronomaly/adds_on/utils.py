import errno

def give_labels(df_tot, num_to_label=200, max_labeled=600):

    features = pd.read_parquet(output_dir + 'FeatureScaler_output.parquet')

    filename = 'ScoreConverter_output.parquet'
    path = output_dir + filename
    
    #TODO add which one to load
    file_list = glob.glob( os.path.join(output_dir, filename.split('.')[0]+'*'))

    if len(file_list)>1:
        #ns was run already it means i need to save it with a number
        counter = num_to_label
        while os.path.exists(path):
            filename = filename.split('.')[0]+f'_{counter}.parquet'
            path = os.path.join(output_dir,filename)
            counter += num_to_label
        #save it with a name that doesnt exist
        print(f'read {filename}')
        #anomalies.to_parquet(path)

        if counter!=None:
            if counter == max_labeled:
                return break

    elif os.path.exists(file_list[0]):
        #first time
        print(f'read {filename}')
    else:
        raise FileNotFoundError(
    errno.ENOENT, os.strerror(errno.ENOENT), filename)

    anomalies = pd.read_parquet(os.path.join(output_dir,filename))

    if 'human_label' not in anomalies.columns:
        anomalies['human_label'] = [-1]*len(anomalies)

    anomalies= anomalies.sort_values('score', ascending=False)
    mask = anomalies['human_label'] == -1
    rows_to_update = anomalies.loc[mask].iloc[:num_to_label].index
    df_tot = df_tot.set_index('KIDS_ID')
    anomalies.loc[rows_to_update, 'human_label'] = df_tot.loc[rows_to_update, 'LABEL_TE']

    ns = NeighbourScore(alpha=0.1, force_rerun=True, output_dir = output_dir)
    features_with_labels = ns.combine_data_frames(features, anomalies)
    final_score = ns.run(features_with_labels)#.drop(['KIDS_ID', 'LABEL_TE'],axis=1))
    anomalies['final_score'] = final_score.trained_score
    num_lab = len(anomalies) - len(anomalies.loc[anomalies['human_label'] == -1])
    print(f'saved as ScoreConverter_{num_lab}_output.parquet')
    anomalies.to_parquet(output_dir + f'ScoreConverter_output_{num_lab}.parquet')




def recall(ml_score, bin=1, column='LABEL_TE', sort_by='score'):

    #ml_score[column] = ml_score['Unnamed: 0'].map(df_tot.set_index('KIDS_ID')[column]).replace(1, 5)
    df_sorted = ml_score.sort_values(sort_by, ascending=False)
    num_elements = np.arange(1, len(df_sorted)+1, bin)
    recalls = []
    for i in num_elements:
        true_labels = df_sorted.iloc[:i][column]
        remaining_lablels = df_sorted.iloc[i:][column]
        TP = len(np.where(true_labels==5)[0])
        FN = len(np.where(remaining_lablels==5)[0])
        recall = TP / (TP+FN)
        
        recalls.append(recall)
    return num_elements, recalls


def tp_norm(ml_score, bin=1, column='LABEL_TE'):

    df_sorted = ml_score.sort_values('score', ascending=False)
    num_elements = np.arange(1, len(df_sorted)+1, bin)
    recalls = []
    for i in num_elements:
        true_labels = df_sorted.iloc[:i][column]
        TP = len(np.where(true_labels==5)[0])
        recall = TP / len(true_labels)
        recalls.append(recall)
    return num_elements, recalls


def give_labels_old(df_tot, filename = 'ml_scores.csv'):

    path = os.path.join(output_dir,filename)
    #read and save for later
    ml_score = pd.read_csv(os.path.join(output_dir,filename))

    counter = 0

    while os.path.exists(path):
        filename = f'ml_scores_{counter}.csv'
        path = os.path.join(output_dir,filename)
        counter += 1
        
    #print()

    if (len(ml_score[ml_score['human_label'] !=-1])  ) != (counter-1 )*100 :
        print(len(ml_score[ml_score['human_label'] !=-1]) - 1 )
        print('something went wrong with file name, not saved')
        return
    

    ml_score.to_csv(path, index=False)
    print('saved as ',filename)

    ml_score_tmp = ml_score.copy()
    ml_score_tmp['LABEL_TE'] = ml_score_tmp['Unnamed: 0'].map(df_tot.set_index('KIDS_ID')['LABEL_TE']).replace(1, 5)
    ml_score_tmp = ml_score_tmp.sort_values('score', ascending=False)
    
    mask = ml_score_tmp['human_label'] == -1
    rows_to_update = ml_score_tmp.loc[mask].iloc[:99].index
    ml_score_tmp.loc[rows_to_update, 'human_label'] = ml_score_tmp.loc[rows_to_update, 'LABEL_TE']
    print('labelled {} obj'.format(len(ml_score_tmp[ml_score_tmp['human_label'] !=-1])))

    ml_score_tmp.to_csv(os.path.join(output_dir,'ml_scores.csv'), index=False)


