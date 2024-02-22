import train
import eval

if __name__ == "__main__":
    run_name = train.main()
    eval.main(run_name)
