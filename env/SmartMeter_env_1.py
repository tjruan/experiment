from __future__ import annotations
from typing import Any, ClassVar
import torch
import random
import numpy as np
import pandas as pd
from Discriminator1 import D1
from Discriminator2 import D2
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU, Box

@env_register
class SmartMeterEnv1(CMDP):
    """Smart Meters Environment"""
    need_auto_reset_wrapper: bool = True   
    need_time_limit_wrapper: bool = True   

    _support_envs: ClassVar[list[str]] = ['SmartMeter1-v0']

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = torch.device(device)
        
        # actor --> ActionScale wrapper --> environment
        self._action_space = Box(low=0, high=1, shape=(1,))
        self._observation_space = Box(low=-1, high=10000, shape=(3,))
        assert isinstance(self._observation_space, Box), 'Only support Box action space.'
        assert isinstance(self._action_space, Box), 'Only support Box observation space.'

        self.trainD_flag = 0  # Flag for network parameter updates
        self.trainD = 1       # Frequency of updating different network parameters
        self.mu = 0           # Mean
        self.sigma = 1        # Standard deviation
        self.D_sm1 = D1()     # Discriminator 1
        self.D_sm2 = D2()     # Discriminator 2
        self.price = [0.101, 0.144, 0.208]  # Real-time electricity prices
        self.C_RB = 6000      # Battery capacity (unit: Wh)
        self.currentC = 0     # Current battery charge (unit: Wh)
        self.RB_max = 3000    # Maximum battery charging power (unit: W)
        self.RB_min = -3000   # Maximum battery discharging power (unit: W)
        self.sample = 120     # Sampling interval (unit: s)
        self.days = 0         # Current number of days
        self.days_max = 1     # Number of days read in one episode
        self.day = 0          # Current date
        self.month = 0        # Current month
        self.price_tag = 0    # Electricity price tag [0, 1, 2]
        self.Usr = 0          # User load
        self.Grid = 0         # Grid load
        self.time = 0         # Seconds [0, 86400]
        self.time_start = 0   # Start time of one day's data (unit: s)
        self.start_hour = 0   # Hour [0, 24]
        self.Grid_max = 6000  # Maximum value of grid load
        self.penalty_coef = 0.2  # Penalty coefficient
        self.penalty = 0      # Penalty for truncating actions that violate realistic conditions
        self.action = 0       # Policy action load value

    def step(
        self,
        Grid: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        # init
        terminated = False
        truncated = False

        self.Grid = np.clip(Grid.item(), 0, 1) * self.Grid_max
        self.action = self.Grid
        # Battery charge/discharge power satisfies the relation: RB_min <= Grid-Usr <= RB_max
        RB = min(max(self.Grid - self.Usr, self.RB_min), self.RB_max)  # unit: w
        
        # For battery storage: 0 <= currentC <= C_RB, adjust for RB
        currentC_temp = self.currentC
        self.currentC = min(max(self.currentC + RB * self.sample / 3600, 0), self.C_RB)
        RB = (self.currentC - currentC_temp) * (3600 / self.sample)
        self.Grid = self.Usr + RB

        # Current time slot appliance state St and previous time slot appliance state St-1
        S_t = self.total[self.time]
        S_t_1 = self.total[self.time - 1]
        
        # Find the marginal distribution randomly sampled St and St-1
        random_index = random.randint(0, len(self.total) - 2)  # Generate random indexes
        random_S_t_1 = self.total[random_index]
        random_S_t = self.total[random_index + 1]

        self.D_sm1.joint_buffer.append((self.Grid, S_t, S_t_1))
        self.D_sm1.marginal_buffer.append((self.Grid, random_S_t, random_S_t_1))
        self.D_sm2.joint_buffer.append((self.Grid, S_t_1))
        self.D_sm2.marginal_buffer.append((self.Grid, random_S_t_1))
        
        D_kl1 = torch.tensor(self.D_sm1.caculate(self.Grid, S_t, S_t_1), dtype=float).detach()
        D_kl2 = torch.tensor(self.D_sm2.caculate(self.Grid, S_t_1), dtype=float).detach()
        cost = float(D_kl1 - D_kl2)
        
        reward = - self.Grid * self.price[self.price_tag] * self.sample / (3600 * 1000) + (self.currentC - currentC_temp) / 1000 * 0.1

        if self.time - self.time_start >= len(self.total) / 2 - 1:
            self.days += 1  
        
        # Discriminator updates after an episode
        if self.days == self.days_max:  #days_max 读完一幕结束
            terminated = True
            self.trainD_flag += 1
            if self.trainD_flag == self.trainD:
                self.trainD_flag = 0
                self.D_sm1.update()
                self.D_sm2.update()
        
        self.time = self.time + 1
        timehour = self.time / 30 % 24
        if timehour <= 7 or timehour >= 19:
            self.price_tag = 0
        elif timehour > 11 and timehour <= 17:
            self.price_tag = 2
        else:
            self.price_tag = 1

        self.Usr = self.userload[self.time]

        state = np.concatenate((np.array([self.price_tag]), np.array([self.Usr / 1000]), np.array([self.currentC / self.C_RB])))
        
        state = torch.as_tensor(state, dtype=torch.float32, device=self._device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self._device)
        cost = torch.as_tensor(cost, dtype=torch.float32, device=self._device)
        terminated = torch.as_tensor(terminated, dtype=torch.float32, device=self._device)
        truncated = torch.as_tensor(truncated, dtype=torch.float32, device=self._device)
        return state, reward, cost, terminated, truncated, {}

    def reset(
        self,
        seed: int = 0,
        options: dict[str, Any] | None = None,
        test = False,
        day = 82
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        
        if seed is not None:
            self.set_seed(seed)
        """Reset the environment"""
        print("========reset=========")

        #Start at zero for testing, random start for training
        if test == True:
            self.time = 0
            self.start_hour = 0
            self.days_max = 1
            self.month = random.randint(6, 9)
            self.day = day # 82
        else:
            self.day = random.randint(0, 119)
            self.time = int(random.uniform(1, 719))
            self.start_hour = random.uniform(0, 23.9)
            self.days_max = 1
            self.month = 6

        self.days = 0 # current days of training, reset if days_max is reached
        self.time_start = self.time

        # The first column of the dataset is the decimal appliance status and the second column is the user load Xt, calculated from the load of each appliance and the
        totalfile1 = "C:/data_" + str(self.day) + ".csv"
        temp1 = pd.read_csv(totalfile1, header=None)
        self.total = temp1.iloc[:, 0].values.tolist() # St
        userload = temp1.iloc[:, 1].values.tolist()   # Xt 
        # Generate lognormally distributed noise data and save to list
        noiseload = list(np.random.lognormal(self.mu, self.sigma, len(userload)))
        self.userload = [x + y for x, y in zip(userload, noiseload)]

        self.Usr = self.userload[self.time]

        timehour = self.time / 30 % 24
        if timehour <= 7 or timehour >= 19:
            self.price_tag = 0
            self.currentC = random.uniform(0, 100)  #0.100
        elif timehour >= 11 and timehour <= 17:
            self.price_tag = 2
            self.currentC = random.uniform(0, 100) #0.100
        else:
            self.price_tag = 1
            self.currentC = random.uniform(0, 100) #0.100
    
        if test == True:
            self.currentC = 0
        state = np.concatenate((np.array([self.price_tag]), np.array([self.Usr / 1000]), np.array([self.currentC / self.C_RB])))
        return torch.as_tensor(state, dtype=torch.float32, device=self._device), {}
    

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

    def sample_action(self) -> torch.Tensor:
        pass

    def render(self) -> Any:
        pass

    def close(self) -> None:
        pass