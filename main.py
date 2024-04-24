import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
from A2C import A2CAgent
from DDPG import DDPGAgent
from plots import plot_durations, plot_losses, plot_rewards
from dm_control import suite
import torch


def main():

    parser = argparse.ArgumentParser(description='Train or evaluate policy on Deepmind Control Suite')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Mode to run the model: train or test.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Path to the dataset file.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--episodes', type=int, default=1, help='Number of epochs to train the model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--save_interval', type=int, default=50, help='Interval to save the model weights.')
    parser.add_argument('--video_filename', type=str, default='evaluation_cartpole.mp4', help='Filename to save the evaluation video.')
    parser.add_argument('--save_dir', type=str, default='results', help='Path to save the model weights.')
    args = parser.parse_args()
    

    #env = suite.load(domain_name="cartpole", task_name="swingup", task_kwargs={'random': args.seed}) 
    #env = suite.load(domain_name="cheetah", task_name="run",task_kwargs={'random': args.seed})
    env = suite.load(domain_name="walker", task_name="walk", task_kwargs={'random': args.seed})
    
    torch.set_default_dtype(torch.float32) 
    
    if args.mode == 'train':

        trainer = A2CAgent(env, lr=args.learning_rate, 
                           gamma=args.gamma, 
                           seed=args.seed, 
                           save_interval=args.save_interval, 
                           save_dir=args.save_dir)
        trainer.load_model("results/a2c_walker_model_200_1.pth")
        trainer.train(num_episodes=args.episodes)
         
        """
        DDPGtrainer = DDPGAgent(env, gamma=args.gamma, 
                                seed=args.seed,
                                actor_lr=args.learning_rate,
                                critic_lr=args.learning_rate,
                                batch_size=128, #256 #1000
                                memory_size=1000000,
                                tau=0.005,
                                save_dir=args.save_dir,
                                save_interval=args.save_interval)
        #DDPGtrainer.load_model("results/ddpg_walker_model_400_1.pth")
        DDPGtrainer.train(args.episodes)
        """
        
    elif args.mode == 'test':

        
        trainer = A2CAgent(env, lr=args.learning_rate, gamma=args.gamma, seed=args.seed)
        trainer.evaluate("results/a2c_walker_model_200_3.pth", video_filename='videos/a2c_walker_evaluation_200.mp4')
        print("Evaluation completed.")
        checkpoint = torch.load("results/a2c_walker_model_200_3.pth")
        rewards = checkpoint['episode_rewards']
        losses = checkpoint['episode_losses']
        plot_rewards(rewards)
        plt.show()
        plot_losses(losses, "Losses")
        plt.show()
        

        """
        trainer = DDPGAgent(env, seed=args.seed)
        trainer.evaluate("results/ddpg_walker_model_400_2.pth", video_filename='videos/ddpg_evaluation_walker_400.mp4')
        print("Evaluation completed.")
        checkpoint = torch.load("results/ddpg_walker_model_400_2.pth")
        rewards = checkpoint['episode_rewards']
        critic_losses = checkpoint['critic_losses']
        actor_losses = checkpoint['actor_losses']
        plot_rewards(rewards)
        plt.show()
        plot_losses(critic_losses, "Critic Losses")
        plt.show()
        plot_losses(actor_losses, "Actor Losses")
        plt.show()
        """
        
if __name__=="__main__":
    main()