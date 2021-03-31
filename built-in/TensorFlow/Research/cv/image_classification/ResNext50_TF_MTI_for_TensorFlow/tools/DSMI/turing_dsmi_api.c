#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <limits.h>

#include <sys/time.h>
#include <sys/ioctl.h>

#include "dsmi_common_interface.h"

#define ERROR_CODE_MAX_NUM (128)
#define BUFF_SIZE (128)
#define MAX_LEN 256
#define MAX_DEVICE_CNT 64
#define ID_LEN 2
#define ERR_LEN 16
#define UTINFO_LEN 32

char* func_dsmi_get_device_health(int device_id)
{
      static char outInfo[MAX_LEN];
      memset(outInfo, '\0', MAX_LEN);
      int  ret = 0;
      unsigned int  phealth = 0;
                                     
      ret = dsmi_get_device_health(device_id,&phealth);//��ѯ�豸���彡��״̬
      if(ret != 0)
      {
          sprintf(outInfo, "func_dsmi_get_device_health execute failed, ret value = %d.", ret);
          return outInfo;
      }

      if(phealth == 0 || phealth == 1)
      {
         sprintf(outInfo, "func_dsmi_get_device_health execute success, ret value = %d, device_health=%u.", ret, phealth);
      }
      else
      {
         sprintf(outInfo, "func_dsmi_get_device_health execute warning, ret value = %d, device_health=%u.", ret, phealth);
      }

      return outInfo;
}

char* func_dsmi_get_device_temperature(int device_id, int check_value)
{
      static char outInfo[MAX_LEN];
      memset(outInfo, '\0', MAX_LEN);
      int  ret = 0;
      int  ptemperature = 0;
                              
      ret = dsmi_get_device_temperature(device_id,&ptemperature);//��ѯоƬ�¶�
      if(ret != 0)
      {
          sprintf(outInfo, "func_dsmi_get_device_temperature execute failed, ret value = %d.", ret);
          return outInfo;
      }

      if(ptemperature >= check_value)
      {
         sprintf(outInfo, "func_dsmi_get_device_temperature execute warning, ret value = %d, device_temperature=%d.", ret, ptemperature);
      }
      else
      {
         sprintf(outInfo, "func_dsmi_get_device_temperature execute success, ret value = %d, device_temperature=%d.", ret, ptemperature);
      }

      return outInfo;
}

char* func_dsmi_get_device_power_info(int device_id, int check_value)
{
      static char outInfo[MAX_LEN];
      memset(outInfo, '\0', MAX_LEN);
      int  ret = 0;
      struct dsmi_power_info_stru pdevice_power_info = {0};

      ret = dsmi_get_device_power_info(device_id, &pdevice_power_info);
      if(ret != 0)
      {
          sprintf(outInfo, "func_dsmi_get_device_power_info execute failed, ret value = %d.", ret);
          return outInfo;
      }

      if(pdevice_power_info.power >= check_value)
      {
         sprintf(outInfo, "func_dsmi_get_device_power_info execute warning, ret value = %d, device_power=%uW.", ret, pdevice_power_info.power);
      }
      else
      {
         sprintf(outInfo, "func_dsmi_get_device_power_info execute success, ret value = %d, device_power=%uW.", ret, pdevice_power_info.power);
      }

      return outInfo;
}


char* func_dsmi_get_device_utilization_rate(int device_id, int device_type, int check_value)
{
      static char outInfo[MAX_LEN];
      memset(outInfo, '\0', MAX_LEN);
      int  ret = 0;
      unsigned  int  putilization_rate = 0;

      char szInfo[UTINFO_LEN];
      memset(szInfo, '\0', UTINFO_LEN);
      switch(device_type)
      {
          case 1:
              strncpy(szInfo, "ddr_utiliza", UTINFO_LEN);
              break;
          case 2:
              strncpy(szInfo, "aicore_utiliza", UTINFO_LEN);
              break;
          case 5:
              strncpy(szInfo, "ddr_bw_utiliza", UTINFO_LEN);
              break;
          case 6:
              strncpy(szInfo, "hbm_utiliza", UTINFO_LEN);
              break;
          case 10:
              strncpy(szInfo, "hbm_bw_utiliza", UTINFO_LEN);
              break;
          default:
              break;
      }

      ret = dsmi_get_device_utilization_rate(device_id,device_type,&putilization_rate);//��ȡDavinciռ����
      if(ret != 0)
      {
          sprintf(outInfo, "func_dsmi_get_device_utilization_rate execute %s failed, ret value = %d.", szInfo, ret);
          return outInfo;
      }

      if(putilization_rate >= check_value)
      {
          sprintf(outInfo, "func_dsmi_get_device_utilization_rate execute warning, ret value = %d, %s=%u%%.", ret, szInfo, putilization_rate);
      }
      else
      {
          sprintf(outInfo, "func_dsmi_get_device_utilization_rate execute success, ret value = %d, %s=%u%%.", ret, szInfo, putilization_rate);
      }

      return outInfo;
}


char* func_dsmi_get_memory_info(int device_id, int check_value)
{
      static char outInfo[MAX_LEN];
      memset(outInfo, '\0', MAX_LEN);
      int  ret = 0;
      struct dsmi_memory_info_stru  pdevice_memory_info = {0};
      ret = dsmi_get_memory_info(device_id,&pdevice_memory_info);//��ȡDDR�ڴ���Ϣ
      if(ret != 0)
      {
          sprintf(outInfo, "func_dsmi_get_memory_info execute failed, ret value = %d.", ret);
          return outInfo;
      }

      if(pdevice_memory_info.freq != check_value)
      {
          sprintf(outInfo, "func_dsmi_get_memory_info execute warning, ret value = %d, ddr_freq:%dMHz, ddr_size:%ldKB, ddr_utiliza:%d%%.", ret, pdevice_memory_info.freq,
              pdevice_memory_info.memory_size, pdevice_memory_info.utiliza);
      }
      else
      {
          sprintf(outInfo, "func_dsmi_get_memory_info execute success, ret value = %d, ddr_freq:%dMHz, ddr_size:%ldKB, ddr_utiliza:%d%%.", ret, pdevice_memory_info.freq,
              pdevice_memory_info.memory_size, pdevice_memory_info.utiliza);
      }

      return outInfo;
}

char* func_dsmi_get_hbm_info(int device_id, int check_value_temp, int check_value_freq)
{
    static char outInfo[MAX_LEN];
    memset(outInfo, '\0', MAX_LEN);
    int  ret = 0;
    struct dsmi_hbm_info_stru  hbm_info;                         

    memset(&hbm_info, 0, sizeof(struct dsmi_hbm_info_stru));
    ret = dsmi_get_hbm_info(device_id,&hbm_info);//��ѯhbm��Ƶ�ʡ���������������Ϣ
    if(ret != 0)
    {
      sprintf(outInfo, "func_dsmi_get_hbm_info execute failed, ret value = %d.", ret);
      return outInfo;
    }

    if(hbm_info.temp > check_value_temp || hbm_info.freq != check_value_freq || hbm_info.memory_usage > hbm_info.memory_size)
    {
        sprintf(outInfo, "func_dsmi_get_hbm_info execute success, ret value = %d, hbm_freq:%dMHz, hbm_size:%ldKB, hbm_usage:%ldKB, hbm_temp:%d, hbm_bw:%d%%.", ret,
        hbm_info.freq, hbm_info.memory_size, hbm_info.memory_usage,hbm_info.temp, hbm_info.bandwith_util_rate);
    }
    else
    {
        sprintf(outInfo, "func_dsmi_get_hbm_info execute success, ret value = %d, hbm_freq:%dMHz, hbm_size:%ldKB, hbm_usage:%ldKB, hbm_temp:%d, hbm_bw:%d%%.", ret,
        hbm_info.freq, hbm_info.memory_size, hbm_info.memory_usage,hbm_info.temp, hbm_info.bandwith_util_rate);
    }

    return outInfo;
}

char* func_dsmi_get_aicore_info(int device_id)
{
      static char outInfo[MAX_LEN];
      memset(outInfo, '\0', MAX_LEN);
      int  ret = 0;
      struct dsmi_aicore_info_stru  aicore_info = {0};                          
      memset(&aicore_info, 0, sizeof(struct dsmi_aicore_info_stru));
      ret = dsmi_get_aicore_info(device_id,&aicore_info);//��ѯaicore��Ƶ�ʡ���������Ϣ
      if(ret != 0)
      {
          sprintf(outInfo, "func_dsmi_get_aicore_info execute failed, ret value = %d.", ret);
          return outInfo;
      }

      if(aicore_info.freq == aicore_info.curfreq)
      {
          sprintf(outInfo, "func_dsmi_get_aicore_info execute success, ret value = %d, freq:%d, cur_frep:%d.", ret,
              aicore_info.freq, aicore_info.curfreq);
      }
      else
      {
          sprintf(outInfo, "func_dsmi_get_aicore_info execute warning, ret value = %d, freq:%d, cur_frep:%d.", ret,
              aicore_info.freq, aicore_info.curfreq);
      }
      return outInfo;
}

char* func_dsmi_get_network_health(int device_id)
{
      static char outInfo[MAX_LEN];
      memset(outInfo, '\0', MAX_LEN);
      int  ret = 0;
      unsigned int  presult = 0;                          

      ret = dsmi_get_network_health(device_id,&presult);//��ѯRoCE���ڼ������IP����ͨ״̬
      if(ret != 0)
      {
          sprintf(outInfo, "func_dsmi_get_network_health execute failed, ret value = %d.", ret);
          return outInfo;
      }
      if(presult == RDFX_DETECT_OK)
      {
          sprintf(outInfo, "func_dsmi_get_network_health execute success, ret value = %d, presult=%u.", ret, presult);
      }
      else
      {
          sprintf(outInfo, "func_dsmi_get_network_health execute warning, ret value = %d, presult=%u.", ret, presult);
      }
      return outInfo;
}

char* func_dsmi_get_phyid_from_logicid(int device_id, int check_value)
{
      static char outInfo[MAX_LEN];
      memset(outInfo, '\0', MAX_LEN);
      int  ret = 0;
      unsigned int  presult = 0;                          

      ret = dsmi_get_phyid_from_logicid(device_id,&presult);
      if(ret != 0)
      {
          sprintf(outInfo, "func_dsmi_get_phyid_from_logicid execute failed, ret value = %d.", ret);
          return outInfo;
      }

      if(presult == check_value)
      {
          sprintf(outInfo, "func_dsmi_get_phyid_from_logicid execute success, ret value = %d, phyid=%u.", ret, presult);
      }
      else
      {
          sprintf(outInfo, "func_dsmi_get_phyid_from_logicid execute warning, ret value = %d, phyid=%u.", ret, presult);
      }

      return outInfo;
}

char* func_dsmi_get_logicid_from_phyid(int phyid, int check_value)
{
      static char outInfo[MAX_LEN];
      memset(outInfo, '\0', MAX_LEN);
      int  ret = 0;
      unsigned int  presult = 0;                          

      ret = dsmi_get_logicid_from_phyid(phyid,&presult);
      if(ret != 0)
      {
          sprintf(outInfo, "func_dsmi_get_logicid_from_phyid execute failed, ret value = %d.", ret);
          return outInfo;
      }

      if(presult == check_value)
      {
          sprintf(outInfo, "func_dsmi_get_logicid_from_phyid execute success, ret value = %d, logicid=%u.", ret, presult);
      }
      else
      {
          sprintf(outInfo, "func_dsmi_get_logicid_from_phyid execute warning, ret value = %d, logicid=%u.", ret, presult);
      }

      return outInfo;
}

char* func_dsmi_enable_container_service()
{
      static char outInfo[MAX_LEN];
      memset(outInfo, '\0', MAX_LEN);
      int  ret = 0;                          

      ret = dsmi_enable_container_service();
      if(ret != 0)
      {
          sprintf(outInfo, "func_dsmi_enable_container_service execute failed, ret value = %d.", ret);
          return outInfo;
      }
      sprintf(outInfo, "func_dsmi_enable_container_service execute success, ret value = %d.", ret);
      return outInfo;
}


char* func_dsmi_get_device_errorcode(int device_id)
{
      static char outInfo[MAX_LEN];
      memset(outInfo, '\0', MAX_LEN);
      int  ret = 0;
      unsigned int perrorcode[ERROR_CODE_MAX_NUM] = {0};
      unsigned int perrorinfo[BUFF_SIZE] = {0};
      int errcnt = 0;

      ret = dsmi_get_device_errorcode(device_id, &errcnt, perrorcode);
      if(ret != 0)
      {
          sprintf(outInfo, "func_dsmi_get_device_errorcode execute failed, ret value = %d.", ret);
          return outInfo;
      }

      char tmpInfo[BUFF_SIZE];
      memset(tmpInfo, '\0', BUFF_SIZE);
      char errInfo[ERR_LEN];
      int i = 0;
      for(i = 0; i < errcnt - 1; i++)
      {
          memset(errInfo, '\0', ERR_LEN);
          sprintf(errInfo, "%u,", perrorcode[i]);
          strncat(tmpInfo, errInfo, ERR_LEN);
      }
      if(i == errcnt - 1)
      {
          memset(errInfo, '\0', ERR_LEN);
          sprintf(errInfo, "%u", perrorcode[i]);
          strncat(tmpInfo, errInfo, ERR_LEN);
      }

      sprintf(outInfo, "func_dsmi_get_device_errorcode execute success, ret value = %d, errorcode=%s.", ret, tmpInfo);

      return outInfo;
}

char* func_dsmi_query_errorstring(int device_id, int errorcode)
{
      static char outInfo[MAX_LEN];
      memset(outInfo, '\0', MAX_LEN);
      int  ret = 0;
      unsigned char perrorinfo[BUFF_SIZE] = {0};

      ret = dsmi_query_errorstring(device_id, errorcode, perrorinfo, BUFF_SIZE);
      if(ret != 0)
      {
         sprintf(outInfo, "func_dsmi_query_errorstring execute errorcode[%d] failed, ret value = %d.", errorcode, ret);
         return outInfo;
      }

      sprintf(outInfo, "func_dsmi_query_errorstring execute warning, ret value = %d, errorinfo=%s.", ret, perrorinfo);

      return outInfo;
}


char* func_dsmi_get_device_list(char* check_value)
{
      static char outInfo[MAX_LEN];
      memset(outInfo, '\0', MAX_LEN);
      int  ret = 0;  
      int device_count = 0;

      ret = dsmi_get_device_count(&device_count);
      if(ret != 0)
      {
          sprintf(outInfo, "func_dsmi_get_device_count execute failed, ret value = %d.", ret);
          return outInfo;
      }
      //sprintf(outInfo, "func_dsmi_get_device_count execute success, ret value = %d, device_count = %d.", ret, device_count);

      int device_id_list[MAX_DEVICE_CNT] = {0};
      ret = dsmi_list_device(device_id_list, device_count);
      if(ret != 0)
      {
          sprintf(outInfo, "func_dsmi_list_device execute failed, ret value = %d.", ret);
          return outInfo;
      }
      
      char tmpInfo[MAX_DEVICE_CNT];
      memset(tmpInfo, '\0', MAX_DEVICE_CNT);
      char idInfo[ID_LEN];
      int i = 0;
      for(i = 0; i < device_count - 1; i++)
      {
          memset(idInfo, '\0', ID_LEN);
          sprintf(idInfo, "%d,", device_id_list[i]);
          strncat(tmpInfo, idInfo, ID_LEN);
      }
      if(i == device_count - 1)
      {
          memset(idInfo, '\0', ID_LEN);
          sprintf(idInfo, "%d", device_id_list[i]);
          strncat(tmpInfo, idInfo, ID_LEN);
      }

      if(check_value == NULL || 0 == strcmp(tmpInfo, check_value) || 0 == strcmp("", check_value))
      {
          sprintf(outInfo, "func_dsmi_list_device execute success, ret value=%d, device_id_list=%s.", ret, tmpInfo);
      }
      else
      {
          sprintf(outInfo, "func_dsmi_list_device execute warning, ret value=%d, device_id_list=%s.", ret, tmpInfo);
      }

      return outInfo;
}


char* func_dsmi_get_device_ip_address(int device_id, char* check_value)
{
      static char outInfo[MAX_LEN];
      memset(outInfo, '\0', MAX_LEN);
      int  ret = 0;  
      ip_addr_t ip_address;
      ip_addr_t mask_address;
      memset(&ip_address, 0, sizeof(ip_addr_t));
      memset(&mask_address, 0, sizeof(ip_addr_t));
      ip_address.ip_type = IPADDR_TYPE_V4; // 0U
      mask_address.ip_type = IPADDR_TYPE_V4; // 0U

      ret = dsmi_get_device_ip_address(device_id, 1, 1, &ip_address, &mask_address);
      if(ret != 0)
      {
          sprintf(outInfo, "func_dsmi_get_device_ip_address execute failed, ret value = %d.", ret);
          return outInfo;
      }

      char ipInfo[UTINFO_LEN];
      memset(ipInfo, '\0', UTINFO_LEN);
      sprintf(ipInfo, "%d.%d.%d.%d", ip_address.u_addr.ip4[0], ip_address.u_addr.ip4[1], ip_address.u_addr.ip4[2], ip_address.u_addr.ip4[3]);

      if(check_value == NULL || 0 == strcmp(ipInfo, check_value) || 0 == strcmp("", check_value))
      {
          sprintf(outInfo, "func_dsmi_get_device_ip_address execute success, ret value = %d, ip_address=%s, mask_address=%d.%d.%d.%d.", ret,
          ipInfo, mask_address.u_addr.ip4[0], mask_address.u_addr.ip4[1], mask_address.u_addr.ip4[2], mask_address.u_addr.ip4[3]);
      }
      else
      {
          sprintf(outInfo, "func_dsmi_get_device_ip_address execute warning, ret value = %d, ip_address=%s, mask_address=%d.%d.%d.%d.", ret,
          ipInfo, mask_address.u_addr.ip4[0], mask_address.u_addr.ip4[1], mask_address.u_addr.ip4[2], mask_address.u_addr.ip4[3]);
      }

      return outInfo;
}

char* func_dsmi_get_chip_info(int device_id)
{
      static char outInfo[MAX_LEN];
      memset(outInfo, '\0', MAX_LEN);
      int  ret = 0;  
      struct dsmi_chip_info_stru chip_info = {0};
      memset(&chip_info, 0, sizeof(struct dsmi_chip_info_stru));

      ret = dsmi_get_chip_info(device_id, &chip_info);
      if(ret != 0)
      {
          sprintf(outInfo, "func_dsmi_get_chip_info execute failed, ret value = %d.", ret);
          return outInfo;
      }

      sprintf(outInfo, "func_dsmi_get_chip_info execute success, ret value = %d, chip_type=%s, chip_name=%s, chip_ver=%s.", ret, chip_info.chip_type, chip_info.chip_name, chip_info.chip_ver);
      return outInfo;
}

char* func_dsmi_get_aicpu_info(int device_id, int check_value)
{
      static char outInfo[MAX_LEN];
      memset(outInfo, '\0', MAX_LEN);
      int  ret = 0;
      struct dsmi_aicpu_info_stru aicpu_info = {0};
      memset(&aicpu_info, 0, sizeof(struct dsmi_aicpu_info_stru));

      ret = dsmi_get_aicpu_info(device_id, &aicpu_info);
      if(ret != 0)
      {
          sprintf(outInfo, "func_dsmi_get_aicpu_info execute failed, ret value = %d.", ret);
          return outInfo;
      }

      char tmpInfo[BUFF_SIZE];
      memset(tmpInfo, '\0', BUFF_SIZE);
      char utiInfo[UTINFO_LEN];
      int i = 0;
      int iWarningFlag = 0;
      for(i = 0; i < aicpu_info.aicpuNum - 1; i++)
      {
          if(aicpu_info.utilRate[i] > check_value)
          {
              iWarningFlag = 1;
          }
          memset(utiInfo, '\0', UTINFO_LEN);
          sprintf(utiInfo, "%u%%;", aicpu_info.utilRate[i]);
          strncat(tmpInfo, utiInfo, UTINFO_LEN);
      }
      if(i == aicpu_info.aicpuNum - 1)
      {
          if(aicpu_info.utilRate[i] > check_value)
          {
              iWarningFlag = 1;
          }
          memset(utiInfo, '\0', UTINFO_LEN);
          sprintf(utiInfo, "%u%%", aicpu_info.utilRate[i]);
          strncat(tmpInfo, utiInfo, UTINFO_LEN);
      }

      if(iWarningFlag)
      {
          sprintf(outInfo, "func_dsmi_get_aicpu_info execute warning, ret value = %d, maxFreq=%u, curFreq=%u, utilRate=%s.", ret, aicpu_info.maxFreq, aicpu_info.curFreq, tmpInfo);
      }
      else
      {
          sprintf(outInfo, "func_dsmi_get_aicpu_info execute success, ret value = %d, maxFreq=%u, curFreq=%u, utilRate=%s.", ret, aicpu_info.maxFreq, aicpu_info.curFreq, tmpInfo);
      }

      return outInfo;
}

