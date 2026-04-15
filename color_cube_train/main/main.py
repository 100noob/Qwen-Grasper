import split_dataset
import train_infer_eval
import export_model

def main():
    # Data Splitting
    split_dataset.move(split_dataset.train_imgs,
                       split_dataset.img_train,
                       split_dataset.lab_train)

    split_dataset.move(split_dataset.val_imgs,
                       split_dataset.img_val,
                       split_dataset.lab_val)

    split_dataset.move(split_dataset.test_imgs,
                       split_dataset.img_test,
                       split_dataset.lab_test)
    
    print("Train:", len(split_dataset.train_imgs))
    print("Val:", len(split_dataset.val_imgs))
    print("Test:",len(split_dataset.test_imgs))

    # train
    train_infer_eval.train()

    # inference
    train_infer_eval.infer_on_val()

    #test
    train_infer_eval.eval_on_test()
    
    # export onnx
    export_model.export_all()

if __name__ == '__main__':
    main() 