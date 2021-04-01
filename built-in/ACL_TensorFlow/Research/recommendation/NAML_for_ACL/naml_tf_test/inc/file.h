#ifndef _FILE_H_
#define _FILE_H_
/**
 * @brief Read data from file
 * @param [in] filePath: file path
 * @param [out] fileSize: file size
 * @return file data
 */
char *ReadFile(const std::string &filePath, size_t &fileSize);

/**
 * @brief Write data to file
 * @param [in] filePath: file path
 * @param [in] buffer: data to write to file
 * @param [in] size: size to write
 * @return write result
 */
bool WriteFile(const std::string &filePath, const void *buffer, size_t size);
extern uint32_t max_filesize;
extern uint32_t max_filecount;
int modify_LogLevel(char* level);
int RenameFile(const char *basePath, char *srcFile, char *desFile);
int contains(const char *basePath, char * fileName, char * string);
int lastlogindex(char **logname);
int **getlog_rename(const char *basePath, char *testcase);
int MoveFile(const char *basePath, char *srcFile, char *desFile);
unsigned long get_file_size(const char *filename);
int find_file(const char *path, const char *filestring, char **findname, uint32_t max_count, uint32_t *count);
int get_pctrace_file_size(const char *path, const char *filestring, uint32_t max_count, uint32_t *filesize, uint32_t *file_count);  
int move_pctrace_file(const char *path, const char *filestring, const char *newstring);
int clear_pctrace_file(const char *path, const char *filestring);
int read_data_file(void *addr, char *path, uint32_t size);
int init_sem(int sem_id,int init_value);
int del_sem(int sem_id);
int sem_p(int sem_id);
int sem_v(int sem_id);
aclError tcSemaphoreWait(int sem_id);
aclError tcSemaphoreRelease(int sem_id);
void acl_logcheck(const char *basePath, char *deslog, char * keyword);
int check_LogLevel(char* logcfgpatch, char* level);
#endif
